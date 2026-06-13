from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
try:
    from huggingface_hub import get_token
except Exception:  # pragma: no cover - compatibility with older hub versions
    get_token = lambda: None  # type: ignore
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


DEFAULT_MODEL_ID = "FarmerlineML/main_twi_TTS"
KEY_CONFIG_FILES = (
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "vocab.json",
)


def _safe_json_load(path: str | Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _summarise_json(name: str, data: dict[str, Any] | None) -> dict[str, str]:
    if not data:
        return {"file": name, "status": "not_json_or_unavailable"}

    interesting_keys = [
        "model_type",
        "architectures",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "sampling_rate",
        "sample_rate",
        "phoneme_language",
        "tokenizer_class",
        "pad_token",
        "unk_token",
        "bos_token",
        "eos_token",
    ]
    out: dict[str, str] = {"file": name, "status": "ok"}
    for key in interesting_keys:
        if key in data:
            out[key] = json.dumps(data[key], ensure_ascii=False)
    return out


def _available_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        return get_token()
    except Exception:
        return None


def _write_rows(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) or ["kind", "file", "status", "notes"]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inspect_model(model_id: str, repo_type: str, output_csv: str | Path) -> Path:
    token = _available_token()
    api = HfApi(token=token)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Inspecting Hugging Face {repo_type}: {model_id}")
    if token:
        print("Hugging Face token detected; authenticated request will be attempted.")
    else:
        print("No Hugging Face token detected; public/unauthenticated request will be attempted.")

    rows: list[dict[str, str]] = []
    try:
        files = api.list_repo_files(model_id, repo_type=repo_type)
    except RepositoryNotFoundError as exc:
        message = (
            "Repository not found or not accessible. This often means the repo is private/gated, "
            "the current environment is not logged in to Hugging Face, the repo_type is wrong, "
            "or the repo_id is misspelled. Run `huggingface-cli login` or set HF_TOKEN, then retry."
        )
        print(f"\nERROR: {message}")
        print(f"Original exception: {exc}")
        rows.append(
            {
                "kind": "access_error",
                "file": "",
                "status": "repository_not_accessible",
                "repo_id": model_id,
                "repo_type": repo_type,
                "token_detected": str(bool(token)),
                "notes": message,
            }
        )
        _write_rows(output_csv, rows)
        print(f"\nSaved diagnostic row to {output_csv}")
        return output_csv
    except HfHubHTTPError as exc:
        message = "Hugging Face Hub request failed. Check authentication, repo_id, repo_type, and network access."
        print(f"\nERROR: {message}")
        print(f"Original exception: {exc}")
        rows.append(
            {
                "kind": "access_error",
                "file": "",
                "status": "hub_http_error",
                "repo_id": model_id,
                "repo_type": repo_type,
                "token_detected": str(bool(token)),
                "notes": repr(exc),
            }
        )
        _write_rows(output_csv, rows)
        print(f"\nSaved diagnostic row to {output_csv}")
        return output_csv

    try:
        info = api.model_info(model_id) if repo_type == "model" else None
    except Exception as exc:
        print(f"Could not fetch model_info: {exc}")
        info = None

    if info is not None:
        print("\nModel metadata")
        print(f"  sha: {getattr(info, 'sha', None)}")
        print(f"  library_name: {getattr(info, 'library_name', None)}")
        print(f"  pipeline_tag: {getattr(info, 'pipeline_tag', None)}")
        print(f"  tags: {getattr(info, 'tags', None)}")

    print("\nFiles")
    for file_name in files:
        print(f"  {file_name}")

    for file_name in files:
        suffix = Path(file_name).suffix.lower()
        rows.append(
            {
                "kind": "file_inventory",
                "file": file_name,
                "extension": suffix,
                "notes": "",
            }
        )

    print("\nConfig/tokenizer summaries")
    for file_name in KEY_CONFIG_FILES:
        if file_name not in files:
            continue
        try:
            local_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                repo_type=repo_type,
                token=token,
            )
            data = _safe_json_load(local_path)
            summary = _summarise_json(file_name, data)
            print(f"  {file_name}: {summary}")
            rows.append({"kind": "json_summary", **{k: str(v) for k, v in summary.items()}})
        except Exception as exc:
            print(f"  {file_name}: unavailable ({exc})")
            rows.append(
                {
                    "kind": "json_summary",
                    "file": file_name,
                    "status": "download_failed",
                    "notes": repr(exc),
                }
            )

    _write_rows(output_csv, rows)
    print(f"\nSaved model inventory to {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the current Farmerline Twi TTS model repository without loading the checkpoint."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    parser.add_argument("--output", default="results/model_file_inventory.csv")
    args = parser.parse_args()

    inspect_model(model_id=args.model_id, repo_type=args.repo_type, output_csv=args.output)


if __name__ == "__main__":
    main()
