from __future__ import annotations

import argparse

from src.data.make_dev_set import save_dev_set
from src.utils.paths import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed Week 1 dev set manifest.")
    parser.add_argument("--config", default="configs/week1_eval.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    out_path = save_dev_set(config)
    print(f"Saved dev set to {out_path}")


if __name__ == "__main__":
    main()
