from __future__ import annotations

import argparse

from src.eval.asr_transcribe import transcribe_audio_manifest
from src.eval.compute_wer import compute_roundtrip_wer, summarise_wer
from src.utils.paths import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASR on baseline TTS audio and compute WER/CER."
    )
    parser.add_argument("--config", default="configs/week1_eval.yaml")
    parser.add_argument(
        "--manifest-with-audio",
        default="data/manifests/dev_set_with_baseline_audio.csv",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    asr_cfg = config["asr"]

    transcripts = transcribe_audio_manifest(
        manifest_with_audio_csv=args.manifest_with_audio,
        output_csv=asr_cfg["output_csv"],
        asr_model_id=asr_cfg["hf_model_id"],
        backend=asr_cfg.get("backend", "qwen2_audio"),
        language=asr_cfg.get("language"),
        task=asr_cfg.get("task"),
        max_new_tokens=asr_cfg.get("max_new_tokens", 256),
    )

    n_rows = len(transcripts)
    n_errors = int((transcripts["asr_error"].fillna("") != "").sum())
    n_empty = int((transcripts["asr_transcript"].fillna("").str.strip() == "").sum())

    print("ASR summary:")
    print(f"  rows: {n_rows}")
    print(f"  rows_with_errors: {n_errors}")
    print(f"  rows_with_empty_transcripts: {n_empty}")

    if n_rows > 0 and n_empty == n_rows:
        raise RuntimeError(
            "All ASR transcripts are empty. "
            "Stopping before WER/CER computation because the result would be invalid."
        )

    wer_df = compute_roundtrip_wer(
        transcripts_csv=asr_cfg["output_csv"],
        output_csv=config["wer"]["output_csv"],
        normalisation_config_path=config["wer"].get("text_normalisation_config"),
    )

    print("WER summary:")
    for key, value in summarise_wer(wer_df).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
