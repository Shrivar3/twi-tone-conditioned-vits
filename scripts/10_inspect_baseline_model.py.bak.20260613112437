from __future__ import annotations

from huggingface_hub import HfApi


def main() -> None:
    model_id = "FarmerlineML/twi-tts-2026"
    api = HfApi()

    print(f"Listing files for {model_id}")
    files = api.list_repo_files(model_id, repo_type="model")

    for file in files:
        print(file)


if __name__ == "__main__":
    main()
