from __future__ import annotations

from typing import Any

from datasets import Audio, DatasetDict, load_dataset


def load_twi_dataset(
    hf_dataset_id: str = "FarmerlineML/Twi_TTS2026_dataset",
    target_sampling_rate: int | None = 22050,
    **load_kwargs: Any,
) -> DatasetDict:
    """Load the Farmerline Twi TTS dataset from Hugging Face.

    Authentication is handled by the Hugging Face CLI or environment variables.
    Run `hf auth login` before using this on a gated/private dataset.
    """
    ds = load_dataset(hf_dataset_id, **load_kwargs)

    if target_sampling_rate is not None:
        for split in ds.keys():
            if "audio" in ds[split].column_names:
                ds[split] = ds[split].cast_column(
                    "audio", Audio(sampling_rate=target_sampling_rate)
                )
    return ds


def split_to_dataframe(ds: DatasetDict, split: str):
    """Convert a split to pandas, avoiding audio decoding where possible."""
    return ds[split].remove_columns(
        [c for c in ["audio"] if c in ds[split].column_names]
    ).to_pandas()
