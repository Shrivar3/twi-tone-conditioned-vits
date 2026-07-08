from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import ensure_dir, ensure_parent, resolve_path


RATING_COLUMNS = [
    "rater_id",
    "utt_id",
    "clip_id",
    "audio_path",
    "naturalness_score",
    "intelligibility_score",
    "tone_accuracy_score",
    "comments",
]

KEY_COLUMNS = ["utt_id", "clip_id", "system"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a blind paired MOS comparison sheet.",
    )
    parser.add_argument(
        "--baseline-manifest",
        default="data/manifests/dev_set_with_baseline_audio.csv",
        help="Baseline synthesis manifest with utt_id, mos_subset, and audio path.",
    )
    parser.add_argument(
        "--tone-manifest",
        default="data/manifests/dev_set_with_tone_on_audio.csv",
        help="Tone-conditioned synthesis manifest with utt_id, mos_subset, and audio path.",
    )
    parser.add_argument(
        "--output-sheet",
        default="results/mos_comparison_rating_sheet.csv",
        help="Blind rating sheet output CSV.",
    )
    parser.add_argument(
        "--output-key",
        default="results/mos_comparison_hidden_key.csv",
        help="Hidden clip-to-system key output CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for system order within each utterance pair.",
    )
    return parser.parse_args()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _filter_mos_subset(df: pd.DataFrame, *, manifest_name: str) -> pd.DataFrame:
    if "mos_subset" not in df.columns:
        raise ValueError(f"{manifest_name} must contain a mos_subset column.")

    return df[df["mos_subset"].map(_truthy)].copy()


def _audio_column(df: pd.DataFrame, *, system: str) -> str:
    preferred = {
        "baseline": [
            "baseline_tts_audio_path",
            "baseline_audio_path",
            "audio_path",
        ],
        "tone": [
            "tone_on_audio_path",
            "tone_vits_audio_path",
            "tone_audio_path",
            "audio_path",
        ],
    }[system]

    for column in preferred:
        if column in df.columns:
            return column

    candidates = [column for column in df.columns if column.endswith("_audio_path")]
    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"Could not infer {system} audio column. "
        f"Expected one of {preferred} or a single *_audio_path column."
    )


def _load_manifest(path: str, *, system: str) -> pd.DataFrame:
    manifest_path = resolve_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    missing = [column for column in ["utt_id", "mos_subset"] if column not in df.columns]
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {missing}")

    audio_column = _audio_column(df, system=system)
    out = _filter_mos_subset(df, manifest_name=str(manifest_path))
    out = out[["utt_id", audio_column]].copy()
    out = out.rename(columns={audio_column: "source_audio_path"})
    out["utt_id"] = out["utt_id"].astype(str)
    out["system"] = system

    if out["utt_id"].duplicated().any():
        duplicates = sorted(out.loc[out["utt_id"].duplicated(), "utt_id"].unique())
        raise ValueError(f"{manifest_path} contains duplicate mos_subset utt_id values: {duplicates}")

    return out


def _resolve_source_audio(raw_path: str, *, manifest_path: str) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path

    manifest_dir_candidate = resolve_path(manifest_path).parent / path
    if manifest_dir_candidate.exists():
        return manifest_dir_candidate

    return resolve_path(path)


def _blind_audio_dir(output_sheet: str) -> Path:
    output_path = ensure_parent(output_sheet)
    return ensure_dir(output_path.with_name("mos_comparison_audio"))


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(resolve_path(".")).as_posix()
    except ValueError:
        return str(path)


def make_blind_mos_comparison_sheet(
    baseline_manifest: str,
    tone_manifest: str,
    output_sheet: str,
    output_key: str,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_df = _load_manifest(baseline_manifest, system="baseline")
    tone_df = _load_manifest(tone_manifest, system="tone")

    baseline_ids = set(baseline_df["utt_id"])
    tone_ids = set(tone_df["utt_id"])
    if baseline_ids != tone_ids:
        missing_tone = sorted(baseline_ids - tone_ids)
        missing_baseline = sorted(tone_ids - baseline_ids)
        raise ValueError(
            "MOS subsets must contain the same utt_id values. "
            f"Missing from tone manifest: {missing_tone}. "
            f"Missing from baseline manifest: {missing_baseline}."
        )

    baseline_by_utt = baseline_df.set_index("utt_id")
    tone_by_utt = tone_df.set_index("utt_id")
    utt_ids = list(baseline_df["utt_id"])
    rng = random.Random(seed)
    audio_dir = _blind_audio_dir(output_sheet)

    rating_rows: list[dict[str, str]] = []
    key_rows: list[dict[str, str]] = []
    clip_index = 1

    for utt_id in utt_ids:
        pair = [
            baseline_by_utt.loc[utt_id].to_dict(),
            tone_by_utt.loc[utt_id].to_dict(),
        ]
        rng.shuffle(pair)

        for item in pair:
            clip_id = f"clip_{clip_index:04d}"
            source_audio = _resolve_source_audio(
                item["source_audio_path"],
                manifest_path=baseline_manifest if item["system"] == "baseline" else tone_manifest,
            )
            if not source_audio.exists():
                raise FileNotFoundError(f"Audio file not found: {source_audio}")

            suffix = source_audio.suffix or ".wav"
            blind_audio_path = audio_dir / f"{clip_id}{suffix}"
            shutil.copyfile(source_audio, blind_audio_path)

            rating_rows.append(
                {
                    "rater_id": "",
                    "utt_id": utt_id,
                    "clip_id": clip_id,
                    "audio_path": _repo_relative(blind_audio_path),
                    "naturalness_score": "",
                    "intelligibility_score": "",
                    "tone_accuracy_score": "",
                    "comments": "",
                }
            )
            key_rows.append(
                {
                    "utt_id": utt_id,
                    "clip_id": clip_id,
                    "system": item["system"],
                }
            )
            clip_index += 1

    rating_df = pd.DataFrame(rating_rows, columns=RATING_COLUMNS)
    key_df = pd.DataFrame(key_rows, columns=KEY_COLUMNS)

    rating_df.to_csv(ensure_parent(output_sheet), index=False)
    key_df.to_csv(ensure_parent(output_key), index=False)

    return rating_df, key_df


def main() -> None:
    args = parse_args()

    rating_df, key_df = make_blind_mos_comparison_sheet(
        baseline_manifest=args.baseline_manifest,
        tone_manifest=args.tone_manifest,
        output_sheet=args.output_sheet,
        output_key=args.output_key,
        seed=args.seed,
    )

    print(f"Wrote blind MOS rating sheet: {ensure_parent(args.output_sheet)}")
    print(f"Wrote hidden key: {ensure_parent(args.output_key)}")
    print(f"Wrote {len(rating_df)} clips across {key_df['utt_id'].nunique()} utterances.")


if __name__ == "__main__":
    main()
