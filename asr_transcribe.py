from __future__ import annotations

import argparse

from src.eval.asr_transcribe import transcribe_audio_manifest
from src.eval.compute_wer import compute_roundtrip_wer, summarise_wer
from src.utils.paths import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR on baseline TTS audio and compute WER/CER.")
    parser.add_argument("--config", default="configs/week1_eval.yaml")
    parser.add_argument("--manifest-with-audio", default="data/manifests/dev_set_with_baseline_audio.csv")
    args = parser.parse_args()

    config = load_yaml(args.config)
    asr_cfg = config["asr"]

    transcripts = transcribe_audio_manifest(
        manifest_with_audio_csv=args.manifest_with_audio,
        output_csv=asr_cfg["output_csv"],
        asr_model_id=asr_cfg["hf_model_id"],
        language=asr_cfg.get("language"),
        task=asr_cfg.get("task", "transcribe"),
    )

    wer_df = compute_roundtrip_wer(
        transcripts_csv=asr_cfg["output_csv"],
        output_csv=config["wer"]["output_csv"],
        normalisation_config_path=config["wer"].get("text_normalisation_config"),
    )

    print("WER summary:")
    for k, v in summarise_wer(wer_df).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
