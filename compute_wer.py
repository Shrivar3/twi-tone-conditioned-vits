from __future__ import annotations

import argparse

from src.tone.annotate_tone import annotate_manifest, make_native_validation_sheet
from src.utils.paths import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate dev set with conservative Akan/Twi tone labels.")
    parser.add_argument("--config", default="configs/week1_eval.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    tone_cfg = config["tone_annotation"]

    tone_df = annotate_manifest(
        input_csv=config["dev_set"]["output_csv"],
        output_csv=tone_cfg["output_csv"],
        unknown_label=tone_cfg.get("unknown_label", "UNK"),
        high_label=tone_cfg.get("high_label", "H"),
        low_label=tone_cfg.get("low_label", "L"),
        falling_label=tone_cfg.get("falling_label", "F"),
    )
    make_native_validation_sheet(tone_df, tone_cfg["validation_sheet_csv"])

    print(f"Saved tone annotations to {tone_cfg['output_csv']}")
    print(f"Saved native validation sheet to {tone_cfg['validation_sheet_csv']}")


if __name__ == "__main__":
    main()
