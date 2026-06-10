from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from pydantic import BaseModel, Field


class ToneItem(BaseModel):
    token: str = Field(description="The original Twi token.")
    tone_sequence: str = Field(
        description=(
            "Candidate lexical tone sequence for this token. Use H for high, "
            "L for low, F for falling, combinations like H-L where needed, "
            "or UNK if tone cannot be reliably inferred."
        )
    )
    confidence: float = Field(
        description="Confidence from 0.0 to 1.0. Use low confidence when uncertain."
    )
    reason: str = Field(
        description="Brief reason for the tone label, especially if uncertain."
    )


class ToneAnnotation(BaseModel):
    items: list[ToneItem]
    sentence_confidence: float = Field(
        description="Overall confidence from 0.0 to 1.0 for the sentence annotation."
    )
    needs_native_review: bool = Field(
        description="True if any token is uncertain or requires native speaker validation."
    )
    overall_comment: str = Field(
        description="Brief comment on ambiguity, uncertainty, or validation needs."
    )


def _extract_json(text: str) -> str:
    """Extract a JSON object from Gemini response text."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in response: {text[:500]}")

    return text[start : end + 1]


def _make_prompt(text: str, tokens: list[str]) -> str:
    return (
        "You are assisting a Twi/Akan speech synthesis research project.\n\n"
        "Task:\n"
        "Annotate the lexical tone of each Twi token in the sentence.\n\n"
        "Tone labels:\n"
        "- H = high tone\n"
        "- L = low tone\n"
        "- F = falling tone\n"
        "- UNK = unknown / cannot be reliably inferred\n\n"
        "Important rules:\n"
        "1. Return exactly one item per input token, in the same order.\n"
        "2. Do not invent confident labels where standard orthography is ambiguous.\n"
        "3. Use UNK when tone cannot be reliably inferred from the written text.\n"
        "4. Candidate labels are allowed, but uncertain labels must have low confidence.\n"
        "5. Set needs_native_review=true whenever any token is uncertain.\n"
        "6. Keep reasons short.\n\n"
        f"Sentence:\n{text}\n\n"
        f"Tokens JSON:\n{json.dumps(tokens, ensure_ascii=False)}\n\n"
        "Return JSON only."
    )


def annotate_text_with_gemini(
    text: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
) -> ToneAnnotation:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY first."
        )

    tokens = text.split()
    prompt = _make_prompt(text=text, tokens=tokens)

    client = genai.Client(api_key=api_key)

    schema = ToneAnnotation.model_json_schema()

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
                "response_format": {
                    "text": {
                        "mime_type": "application/json",
                        "schema": schema,
                    }
                },
            },
        )
    except Exception as structured_error:
        print(
            "Warning: structured-output call failed. "
            f"Retrying with plain JSON prompt. Error: {structured_error}"
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt + "\nReturn only valid JSON matching the requested schema.",
        )

    json_text = _extract_json(response.text)
    annotation = ToneAnnotation.model_validate_json(json_text)

    if len(annotation.items) != len(tokens):
        annotation.needs_native_review = True
        annotation.overall_comment = (
            f"Token count mismatch: expected {len(tokens)}, got "
            f"{len(annotation.items)}. Native review required. "
            + annotation.overall_comment
        )

    return annotation


def _flatten_annotation(annotation: ToneAnnotation) -> dict[str, Any]:
    return {
        "gemini_tokens": " ".join(item.token for item in annotation.items),
        "gemini_tone_sequence": " ".join(item.tone_sequence for item in annotation.items),
        "gemini_token_confidences": " ".join(
            f"{item.confidence:.3f}" for item in annotation.items
        ),
        "gemini_sentence_confidence": f"{annotation.sentence_confidence:.3f}",
        "gemini_needs_native_review": annotation.needs_native_review,
        "gemini_overall_comment": annotation.overall_comment,
        "gemini_items_json": json.dumps(
            [item.model_dump() for item in annotation.items],
            ensure_ascii=False,
        ),
    }


def annotate_csv_with_gemini(
    input_csv: str | Path,
    output_csv: str | Path,
    text_column: str = "text",
    model: str = "gemini-2.5-flash",
    limit: int | None = None,
    sleep_seconds: float = 1.0,
    resume: bool = True,
    temperature: float = 0.1,
) -> Path:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if resume and output_csv.exists():
        with output_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("utt_id"):
                    done_ids.add(row["utt_id"])

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        base_fieldnames = reader.fieldnames or []

    extra_fieldnames = [
        "gemini_model",
        "gemini_status",
        "gemini_error",
        "gemini_tokens",
        "gemini_tone_sequence",
        "gemini_token_confidences",
        "gemini_sentence_confidence",
        "gemini_needs_native_review",
        "gemini_overall_comment",
        "gemini_items_json",
    ]

    fieldnames = base_fieldnames + [
        field for field in extra_fieldnames if field not in base_fieldnames
    ]

    write_header = not output_csv.exists() or not resume

    processed_now = 0

    with output_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for row in rows:
            utt_id = row.get("utt_id", "")

            if resume and utt_id in done_ids:
                continue

            if limit is not None and (len(done_ids) + processed_now) >= limit:
                break

            text = row.get(text_column, "")
            out = dict(row)
            out["gemini_model"] = model

            try:
                print(f"Annotating {utt_id}: {text[:80]}")
                annotation = annotate_text_with_gemini(
                    text=text,
                    model=model,
                    temperature=temperature,
                )
                out.update(_flatten_annotation(annotation))
                out["gemini_status"] = "ok"
                out["gemini_error"] = ""
            except Exception as exc:
                out["gemini_status"] = "error"
                out["gemini_error"] = repr(exc)
                out["gemini_tokens"] = ""
                out["gemini_tone_sequence"] = ""
                out["gemini_token_confidences"] = ""
                out["gemini_sentence_confidence"] = ""
                out["gemini_needs_native_review"] = True
                out["gemini_overall_comment"] = "Gemini annotation failed."
                out["gemini_items_json"] = ""

            writer.writerow(out)
            f.flush()

            processed_now += 1
            if utt_id:
                done_ids.add(utt_id)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    print(f"Saved Gemini annotations to {output_csv}")
    return output_csv
