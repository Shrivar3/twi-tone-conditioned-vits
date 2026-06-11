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


def summarise_wer(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    wers = []
    cers = []

    for row in rows:
        wer = _to_float(row.get("wer", ""))
        cer = _to_float(row.get("cer", ""))

        if wer is not None:
            wers.append(wer)
        if cer is not None:
            cers.append(cer)

    if not wers:
        raise RuntimeError("No valid WER values found.")

    summary = [
        ("n_samples", len(wers)),
        ("mean_wer", statistics.mean(wers)),
        ("median_wer", statistics.median(wers)),
        ("std_wer", statistics.stdev(wers) if len(wers) > 1 else 0.0),
    ]

    if cers:
        summary.extend(
            [
                ("mean_cer", statistics.mean(cers)),
                ("median_cer", statistics.median(cers)),
                ("std_cer", statistics.stdev(cers) if len(cers) > 1 else 0.0),
            ]
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary)

    print(f"Saved WER summary to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise baseline WER results.")
    parser.add_argument("--input", default="results/baseline_wer.csv")
    parser.add_argument("--output", default="results/baseline_wer_summary.csv")
    args = parser.parse_args()

    summarise_wer(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
