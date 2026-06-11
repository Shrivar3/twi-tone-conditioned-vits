from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "utt_id",
    "token_index",
    "token",
    "context_text",
    "candidate_tone",
    "candidate_confidence",
    "candidate_source",
    "candidate_reason",
    "native_tone_label",
    "native_confidence",
    "review_status",
    "reviewer_id",
    "reviewer_notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _load_gemini_by_utt(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        utt_id = row.get("utt_id", "")
        if utt_id and row.get("gemini_status", "ok") == "ok":
            out[utt_id] = row
    return out


def _items_from_gemini_row(row: dict[str, str]) -> list[dict[str, Any]]:
    raw = row.get("gemini_items_json", "")
    if raw:
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    tokens = row.get("gemini_tokens", "").split()
    tones = row.get("gemini_tone_sequence", "").split()
    confs = row.get("gemini_token_confidences", "").split()

    items: list[dict[str, Any]] = []
    for i, token in enumerate(tokens):
        items.append(
            {
                "token": token,
                "tone_sequence": tones[i] if i < len(tones) else "UNK",
                "confidence": confs[i] if i < len(confs) else "",
                "reason": "Gemini candidate label.",
            }
        )
    return items


def _make_rows(
    dev_rows: list[dict[str, str]],
    gemini_by_utt: dict[str, dict[str, str]],
    limit_utterances: int | None,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    selected = dev_rows[:limit_utterances] if limit_utterances else dev_rows

    for dev_row in selected:
        utt_id = dev_row.get("utt_id", "")
        text = dev_row.get("text", "")
        gemini_row = gemini_by_utt.get(utt_id)

        if gemini_row:
            items = _items_from_gemini_row(gemini_row)
            source = "gemini"
        else:
            items = [
                {
                    "token": token,
                    "tone_sequence": "UNK",
                    "confidence": "0.000",
                    "reason": "No Gemini candidate available; native review required.",
                }
                for token in text.split()
            ]
            source = "blank_native_review"

        for idx, item in enumerate(items):
            out_rows.append(
                {
                    "utt_id": utt_id,
                    "token_index": idx,
                    "token": item.get("token", ""),
                    "context_text": text,
                    "candidate_tone": item.get("tone_sequence", "UNK"),
                    "candidate_confidence": _safe_float(item.get("confidence", "")),
                    "candidate_source": source,
                    "candidate_reason": str(item.get("reason", "")).replace("\n", " ").replace("\r", " "),
                    "native_tone_label": "",
                    "native_confidence": "",
                    "review_status": "pending",
                    "reviewer_id": "",
                    "reviewer_notes": "",
                }
            )
    return out_rows


def prepare_native_validation_pack(
    dev_csv: Path,
    gemini_csv: Path,
    output_csv: Path,
    limit_utterances: int | None = None,
) -> Path:
    dev_rows = _read_csv(dev_csv)
    if not dev_rows:
        raise RuntimeError(f"No rows found in {dev_csv}")

    gemini_by_utt = _load_gemini_by_utt(gemini_csv)
    rows = _make_rows(dev_rows, gemini_by_utt, limit_utterances)
    _write_csv(output_csv, rows, OUTPUT_COLUMNS)

    n_utt = len({row["utt_id"] for row in rows})
    n_with_gemini = len({row["utt_id"] for row in rows if row["candidate_source"] == "gemini"})

    print(f"Saved native validation sheet to {output_csv}")
    print(f"Utterances included: {n_utt}")
    print(f"Utterances with Gemini candidates: {n_with_gemini}")
    print(f"Token rows for review: {len(rows)}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a token-level native-speaker tone validation sheet.")
    parser.add_argument("--dev-csv", default="data/manifests/dev_set.csv")
    parser.add_argument("--gemini-csv", default="data/manifests/gemini_tone_annotated_dev.csv")
    parser.add_argument("--output", default="data/manifests/native_validation_token_sheet.csv")
    parser.add_argument(
        "--limit-utterances",
        type=int,
        default=None,
        help="Optional number of utterances to include for a pilot review.",
    )
    args = parser.parse_args()

    prepare_native_validation_pack(
        dev_csv=Path(args.dev_csv),
        gemini_csv=Path(args.gemini_csv),
        output_csv=Path(args.output),
        limit_utterances=args.limit_utterances,
    )


if __name__ == "__main__":
    main()
