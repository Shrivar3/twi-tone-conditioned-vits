from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def _to_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarise_mos(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    valid_rows = []
    naturalness = []
    intelligibility = []

    for row in rows:
        nat = _to_float(row.get("naturalness_score", ""))
        intel = _to_float(row.get("intelligibility_score", ""))

        if nat is None or intel is None:
            continue

        valid_rows.append(row)
        naturalness.append(nat)
        intelligibility.append(intel)

    if not valid_rows:
        raise RuntimeError("No valid MOS rows found. Fill in the raw ratings first.")

    rater_ids = {row["rater_id"] for row in valid_rows if row.get("rater_id")}
    utt_ids = {row["utt_id"] for row in valid_rows if row.get("utt_id")}

    summary = [
        ("n_ratings", len(valid_rows)),
        ("n_raters", len(rater_ids)),
        ("n_samples", len(utt_ids)),
        ("mean_naturalness", statistics.mean(naturalness)),
        ("std_naturalness", statistics.stdev(naturalness) if len(naturalness) > 1 else 0.0),
        ("mean_intelligibility", statistics.mean(intelligibility)),
        ("std_intelligibility", statistics.stdev(intelligibility) if len(intelligibility) > 1 else 0.0),
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary)

    print(f"Saved MOS summary to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise MOS ratings.")
    parser.add_argument("--input", default="results/baseline_mos_raw.csv")
    parser.add_argument("--output", default="results/baseline_mos_summary.csv")
    args = parser.parse_args()

    summarise_mos(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
