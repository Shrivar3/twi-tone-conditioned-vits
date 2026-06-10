from __future__ import annotations

import argparse

from src.eval.mos_form_export import make_mos_template
from src.eval.synth_baseline import synthesise_manifest
from src.utils.paths import ensure_parent, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline TTS on the fixed dev set.")
    parser.add_argument("--config", default="configs/week1_eval.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    tts_cfg = config["baseline_tts"]
    dev_csv = config["dev_set"]["output_csv"]

    synth_df = synthesise_manifest(
        manifest_csv=dev_csv,
        model_id=tts_cfg["hf_model_id"],
        output_dir=tts_cfg["output_dir"],
        max_samples=tts_cfg.get("max_samples"),
        mos_subset_only=bool(tts_cfg.get("synthesise_mos_subset_only", False)),
    )

    manifest_with_audio = ensure_parent("data/manifests/dev_set_with_baseline_audio.csv")
    synth_df.to_csv(manifest_with_audio, index=False)
    make_mos_template(str(manifest_with_audio), config["mos"]["template_csv"])

    print(f"Saved synthesis manifest to {manifest_with_audio}")
    print(f"Saved MOS template to {config['mos']['template_csv']}")


if __name__ == "__main__":
    main()
