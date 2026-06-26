# scripts/16_smoke_test_tone_conditioned_vits.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from scipy.io.wavfile import write
from transformers import AutoTokenizer, set_seed

from src.modeling.tone_conditioned_vits import ToneConditionedVitsModel
from src.tone.token_tone_alignment import (
    build_token_tone_encoding,
    build_tone_conditioned_inputs,
)
from src.utils.paths import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test tone-conditioned VITS inference path.",
    )
    parser.add_argument(
        "--model-id",
        default="FarmerlineML/main_twi_TTS",
        help="HF model id or local model path.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Input Twi text.",
    )
    parser.add_argument(
        "--tone-sequence",
        default=None,
        help=(
            "Optional token-level tone sequence, e.g. 'H L H'. "
            "Use the same token order as the whitespace/token annotation output."
        ),
    )
    parser.add_argument(
        "--output-wav",
        default="outputs/tone_conditioned_smoke_test.wav",
        help="Output WAV path. outputs/ should remain gitignored.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--tone-embedding-scale",
        type=float,
        default=0.0,
        help=(
            "Use 0.0 for architecture smoke test. "
            "Use >0 only after training tone embeddings."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    encoding_report = build_token_tone_encoding(
        text=args.text,
        tokenizer=tokenizer,
        tone_sequence=args.tone_sequence,
    )

    print("Tone alignment report")
    print("---------------------")
    print(f"normalised_text: {encoding_report.normalised_text_for_alignment}")
    print(f"input_id_len:    {len(encoding_report.input_ids)}")
    print(f"tone_id_len:     {len(encoding_report.tone_ids)}")
    print(f"status:          {encoding_report.alignment_status}")
    print(f"summary:         {encoding_report.tone_summary}")

    model = ToneConditionedVitsModel.from_pretrained(
        args.model_id,
        tone_embedding_scale=args.tone_embedding_scale,
    )
    model.to(args.device)
    model.eval()

    inputs = build_tone_conditioned_inputs(
        text=args.text,
        tokenizer=tokenizer,
        tone_sequence=args.tone_sequence,
        device=args.device,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    waveform = outputs.waveform[0].detach().cpu().numpy()

    output_path = ensure_parent(args.output_wav)
    write(output_path, model.config.sampling_rate, waveform)

    print(f"Wrote smoke-test audio to {Path(output_path)}")
    print(f"Sampling rate: {model.config.sampling_rate}")
    print(f"Waveform shape: {waveform.shape}")


if __name__ == "__main__":
    main()
