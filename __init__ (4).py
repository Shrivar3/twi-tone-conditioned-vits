from __future__ import annotations

import pandas as pd

from src.utils.paths import ensure_parent


def make_mos_template(
    manifest_with_audio_csv: str,
    output_csv: str,
    mos_subset_only: bool = True,
) -> pd.DataFrame:
    """Create a simple CSV to import into a form/spreadsheet for MOS ratings."""
    df = pd.read_csv(manifest_with_audio_csv)
    if mos_subset_only and "mos_subset" in df.columns:
        df = df[df["mos_subset"].astype(bool)].copy()

    out = pd.DataFrame(
        {
            "utt_id": df["utt_id"],
            "text": df["text"],
            "audio_path_or_url": df.get("baseline_tts_audio_path", ""),
            "rater_id": "",
            "naturalness_mos_1_to_5": "",
            "intelligibility_mos_1_to_5": "",
            "comments": "",
        }
    )
    out_path = ensure_parent(output_csv)
    out.to_csv(out_path, index=False)
    return out


def summarise_mos(completed_mos_csv: str) -> dict:
    df = pd.read_csv(completed_mos_csv)
    nat = pd.to_numeric(df["naturalness_mos_1_to_5"], errors="coerce")
    intel = pd.to_numeric(df["intelligibility_mos_1_to_5"], errors="coerce")
    return {
        "n_ratings": int(len(df)),
        "n_raters": int(df["rater_id"].nunique()) if "rater_id" in df.columns else None,
        "naturalness_mean": float(nat.mean()),
        "naturalness_std": float(nat.std()),
        "intelligibility_mean": float(intel.mean()),
        "intelligibility_std": float(intel.std()),
    }
