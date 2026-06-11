from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = ["metric", "value"]
DISAGREEMENT_COLUMNS = [
    "utt_id",
    "token_index",
    "token",
    "context_text",
    "candidate_tone",
    "native_tone_label",
    "native_confidence",
    "reviewer_id",
    "reviewer_notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalise_label(label: str) -> str:
    return label.strip().upper().replace(" ", "")


def _is_reviewed(row: dict[str, str]) -> bool:
    label = row.get("native_tone_label", "").strip()
    status = row.get("review_status", "").strip().lower()
    return bool(label) and status in {"reviewed", "uncertain"}


def summarise_native_validation(input_csv: Path, summary_csv: Path, disagreements_csv: Path) -> None:
    rows = _read_csv(input_csv)
    reviewed = [row for row in rows if _is_reviewed(row)]

    status_counts = Counter(row.get("review_status", "") for row in rows)
    source_counts = Counter(row.get("candidate_source", "") for row in rows)
    native_label_counts = Counter(_normalise_label(row.get("native_tone_label", "")) for row in reviewed)

    comparable = []
    disagreements = []

    for row in reviewed:
        candidate = _normalise_label(row.get("candidate_tone", ""))
        native = _normalise_label(row.get("native_tone_label", ""))

        if not candidate or candidate == "UNK" or not native or native == "UNK":
            continue

        comparable.append(row)
        if candidate != native:
            disagreements.append(
                {
                    "utt_id": row.get("utt_id", ""),
                    "token_index": row.get("token_index", ""),
                    "token": row.get("token", ""),
                    "context_text": row.get("context_text", ""),
                    "candidate_tone": row.get("candidate_tone", ""),
                    "native_tone_label": row.get("native_tone_label", ""),
                    "native_confidence": row.get("native_confidence", ""),
                    "reviewer_id": row.get("reviewer_id", ""),
                    "reviewer_notes": row.get("reviewer_notes", ""),
                }
            )

    n_total = len(rows)
    n_reviewed = len(reviewed)
    n_comparable = len(comparable)
    n_disagree = len(disagreements)
    n_agree = n_comparable - n_disagree
    agreement_rate = n_agree / n_comparable if n_comparable else 0.0
    review_completion = n_reviewed / n_total if n_total else 0.0

    summary_rows = [
        {"metric": "n_total_token_rows", "value": n_total},
        {"metric": "n_reviewed_token_rows", "value": n_reviewed},
        {"metric": "review_completion_rate", "value": f"{review_completion:.4f}"},
        {"metric": "n_comparable_candidate_native_rows", "value": n_comparable},
        {"metric": "n_candidate_native_agreements", "value": n_agree},
        {"metric": "n_candidate_native_disagreements", "value": n_disagree},
        {"metric": "candidate_native_agreement_rate", "value": f"{agreement_rate:.4f}"},
    ]

    for status, count in sorted(status_counts.items()):
        summary_rows.append({"metric": f"review_status_{status or 'blank'}", "value": count})
    for source, count in sorted(source_counts.items()):
        summary_rows.append({"metric": f"candidate_source_{source or 'blank'}", "value": count})
    for label, count in sorted(native_label_counts.items()):
        summary_rows.append({"metric": f"native_label_{label or 'blank'}", "value": count})

    _write_csv(summary_csv, summary_rows, SUMMARY_COLUMNS)
    _write_csv(disagreements_csv, disagreements, DISAGREEMENT_COLUMNS)

    print(f"Saved summary to {summary_csv}")
    print(f"Saved disagreements to {disagreements_csv}")
    print(f"Reviewed rows: {n_reviewed}/{n_total}")
    print(f"Candidate/native agreement rate: {agreement_rate:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise native-speaker tone validation results.")
    parser.add_argument(
        "--input",
        default="data/manifests/native_validation_token_sheet_completed.csv",
        help="Completed native validation token sheet.",
    )
    parser.add_argument("--summary-output", default="results/native_validation_summary.csv")
    parser.add_argument("--disagreements-output", default="results/native_validation_disagreements.csv")
    args = parser.parse_args()

    summarise_native_validation(
        input_csv=Path(args.input),
        summary_csv=Path(args.summary_output),
        disagreements_csv=Path(args.disagreements_output),
    )


if __name__ == "__main__":
    main()
