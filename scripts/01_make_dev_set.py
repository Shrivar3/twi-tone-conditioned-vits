from __future__ import annotations

import argparse
import os
import sys

import yaml

from src.data.make_dev_set import save_dev_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed Week 1 dev set manifest.")
    parser.add_argument("--config", default="configs/week1_eval.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_path = save_dev_set(config)
    print(f"Saved dev set to {out_path}")

    # In Codespaces, Hugging Face Datasets / PyArrow / Torch-related extension
    # modules can occasionally crash during Python interpreter shutdown after
    # streaming audio metadata. The manifest has already been written, so we
    # flush and exit cleanly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
