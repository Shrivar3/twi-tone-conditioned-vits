from __future__ import annotations

from typing import Any

from datasets import Audio, load_dataset


def load_twi_dataset(
    hf_dataset_id: str = "FarmerlineML/Twi_TTS2026_dataset",
    split: str | None = None,
    target_sampling_rate: int | None = 22050,
    streaming: bool = False,
    **load_kwargs: Any,
):
    """Load the Farmerline Twi TTS dataset from Hugging Face.

    Important:
    - For Codespaces, prefer split="test" and streaming=True.
    - Loading the full dataset downloads many large Parquet shards and will
      usually exceed Codespaces disk space.
    """
    if split is not None:
        load_kwargs["split"] = split
    if streaming:
        load_kwargs["streaming"] = True

    ds = load_dataset(hf_dataset_id, **load_kwargs)

    # Do not cast audio in streaming mode. The Week 1 dev-set manifest only
    # needs text and duration metadata, not decoded audio.
    if target_sampling_rate is not None and not streaming:
        if hasattr(ds, "keys"):
            for split_name in ds.keys():
                if "audio" in ds[split_name].column_names:
                    ds[split_name] = ds[split_name].cast_column(
                        "audio", Audio(sampling_rate=target_sampling_rate)
                    )
        elif hasattr(ds, "column_names") and "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio(sampling_rate=target_sampling_rate))

    return ds


def split_to_dataframe(ds, split: str | None = None):
    """Convert a non-streaming split to pandas, avoiding audio decoding."""
    if split is not None and hasattr(ds, "keys"):
        ds = ds[split]
    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])
    return ds.to_pandas()
