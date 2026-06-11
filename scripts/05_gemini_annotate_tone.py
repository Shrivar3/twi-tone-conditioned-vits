from __future__ import annotations

import argparse

from src.tone.gemini_tone_annotator import annotate_csv_with_gemini


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use Gemini to create candidate Twi/Akan tone annotations."
    )
    parser.add_argument(
        "--input",
        default="data/manifests/dev_set.csv",
        help="Input CSV with a text column.",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/gemini_tone_annotated_dev.csv",
        help="Output CSV for Gemini candidate tone annotations.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of the text column in the input CSV.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        help="Gemini model name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new rows to annotate in this run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between API calls.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Gemini generation temperature.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing output CSV by skipping existing utt_id values.",
    )

    args = parser.parse_args()

    annotate_csv_with_gemini(
        input_csv=args.input,
        output_csv=args.output,
        text_column=args.text_column,
        model=args.model,
        limit=args.limit,
        sleep_seconds=args.sleep,
        resume=args.resume,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
