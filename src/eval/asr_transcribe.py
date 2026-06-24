from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils.paths import ensure_parent


QWEN2_AUDIO_SYSTEM_PROMPT = (
    "You are a Twi Akan speech recognition system. "
    "Transcribe the audio exactly as spoken in Twi. "
    "Return only the Twi transcript, nothing else."
)


def _looks_like_qwen2_audio(model_id: str, backend: str | None = None) -> bool:
    if backend and backend.lower() == "qwen2_audio":
        return True

    lowered = model_id.lower()
    return "qwen2audio" in lowered or "qwen2-audio" in lowered


def _load_qwen2_audio_asr(asr_model_id: str) -> tuple[Any, Any, Any, int]:
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen2-Audio ASR requires a GPU for this project. "
            "No CUDA device was detected."
        )

    processor = AutoProcessor.from_pretrained(
        asr_model_id,
        trust_remote_code=True,
    )

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        asr_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    sampling_rate = processor.feature_extractor.sampling_rate
    device = next(model.parameters()).device

    return processor, model, device, sampling_rate


def _transcribe_qwen2_audio(
    audio_path: str,
    processor: Any,
    model: Any,
    device: Any,
    sampling_rate: int,
    max_new_tokens: int = 256,
) -> str:
    import librosa
    import torch

    audio, _ = librosa.load(audio_path, sr=sampling_rate)

    conversation = [
        {
            "role": "system",
            "content": QWEN2_AUDIO_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio_url": audio_path,
                },
                {
                    "type": "text",
                    "text": "Transcribe this Twi audio exactly.",
                },
            ],
        },
    ]

    prompt_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    try:
        inputs = processor(
            text=prompt_text,
            audio=audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
    except TypeError:
        inputs = processor(
            text=prompt_text,
            audios=[audio],
            return_tensors="pt",
            padding=True,
        )

    inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]

    transcript = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return transcript.strip()


def _load_pipeline_asr(asr_model_id: str):
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "automatic-speech-recognition",
        model=asr_model_id,
        device=device,
    )


def transcribe_audio_manifest(
    manifest_with_audio_csv: str,
    output_csv: str,
    asr_model_id: str = "FarmerlineML/twi-asr-qwen2audio-merged",
    backend: str | None = "qwen2_audio",
    language: str | None = None,
    task: str | None = None,
    max_new_tokens: int = 256,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_with_audio_csv)

    if "baseline_tts_audio_path" not in df.columns:
        raise ValueError("Manifest must contain baseline_tts_audio_path")

    use_qwen2_audio = _looks_like_qwen2_audio(asr_model_id, backend=backend)

    if use_qwen2_audio:
        processor, model, device, sampling_rate = _load_qwen2_audio_asr(asr_model_id)
        pipeline_asr = None
    else:
        processor = None
        model = None
        device = None
        sampling_rate = None
        pipeline_asr = _load_pipeline_asr(asr_model_id)

    rows: list[dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running ASR"):
        audio_path = str(row["baseline_tts_audio_path"])

        try:
            if not Path(audio_path).exists():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}. "
                    "If this is a fresh GPU instance, rerun baseline TTS first."
                )

            if use_qwen2_audio:
                transcript = _transcribe_qwen2_audio(
                    audio_path=audio_path,
                    processor=processor,
                    model=model,
                    device=device,
                    sampling_rate=sampling_rate,
                    max_new_tokens=max_new_tokens,
                )
            else:
                generate_kwargs = {}

                if language:
                    generate_kwargs["language"] = language

                if task:
                    generate_kwargs["task"] = task

                result = pipeline_asr(
                    audio_path,
                    generate_kwargs=generate_kwargs if generate_kwargs else None,
                )
                transcript = result.get("text", "") if isinstance(result, dict) else str(result)

            error = ""

        except Exception as exc:
            transcript = ""
            error = repr(exc)

        out = row.to_dict()
        out["asr_model_id"] = asr_model_id
        out["asr_backend"] = backend
        out["asr_transcript"] = transcript
        out["asr_error"] = error
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = ensure_parent(output_csv)
    out_df.to_csv(out_path, index=False)

    n_errors = int((out_df["asr_error"].fillna("") != "").sum())
    n_empty = int((out_df["asr_transcript"].fillna("").str.strip() == "").sum())

    print(f"Saved ASR transcripts to {out_path}")
    print(f"Rows: {len(out_df)}")
    print(f"Rows with ASR errors: {n_errors}")
    print(f"Rows with empty transcripts: {n_empty}")

    if len(out_df) > 0 and n_empty == len(out_df):
        print(
            "WARNING: all ASR transcripts are empty. "
            "Do not treat the resulting WER/CER as valid."
        )

    return out_df
