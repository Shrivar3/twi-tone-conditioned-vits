{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Dataset audit\n",
        "\n",
        "Use this notebook to inspect the Hugging Face dataset before building the Week 1 dev set.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from src.data.load_hf_dataset import load_twi_dataset, split_to_dataframe\n",
        "\n",
        "ds = load_twi_dataset(target_sampling_rate=None)\n",
        "ds\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "\n",
        "summaries = []\n",
        "for split in ds.keys():\n",
        "    df = split_to_dataframe(ds, split)\n",
        "    summary = {\n",
        "        'split': split,\n",
        "        'n_rows': len(df),\n",
        "        'columns': list(df.columns),\n",
        "    }\n",
        "    if 'duration' in df.columns:\n",
        "        summary.update({\n",
        "            'duration_hours': df['duration'].sum() / 3600,\n",
        "            'duration_min': df['duration'].min(),\n",
        "            'duration_mean': df['duration'].mean(),\n",
        "            'duration_max': df['duration'].max(),\n",
        "        })\n",
        "    if 'text' in df.columns:\n",
        "        summary.update({\n",
        "            'empty_text': df['text'].isna().sum() + (df['text'].astype(str).str.strip() == '').sum(),\n",
        "            'duplicate_texts': df['text'].duplicated().sum(),\n",
        "        })\n",
        "    summaries.append(summary)\n",
        "\n",
        "pd.DataFrame(summaries)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "df = split_to_dataframe(ds, 'test')\n",
        "df.head()\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "pygments_lexer": "ipython3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}