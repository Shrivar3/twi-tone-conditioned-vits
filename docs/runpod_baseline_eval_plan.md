# RunPod Baseline Evaluation Plan

## Purpose

Run baseline TTS and round-trip ASR WER for the current Farmerline checkpoint.

## Why RunPod

Codespaces should not be used for baseline TTS/ASR because the full environment downloads large Torch/CUDA packages and model/data files. RunPod/A100 is the intended compute environment for heavy inference.

## Inputs

- `data/manifests/dev_set.csv`
- Baseline model: `FarmerlineML/twi-tts-2026`
- Dataset: `FarmerlineML/Twi_TTS2026_dataset`

## Expected Outputs

Generated baseline audio:

- `outputs/baseline_tts/wavs/dev_0000.wav`
- `outputs/baseline_tts/wavs/dev_0001.wav`
- etc.

Baseline TTS manifest:

- `outputs/baseline_tts/baseline_tts_manifest.csv`

Round-trip ASR WER:

- `results/baseline_wer.csv`
- `results/baseline_wer_summary.csv`

MOS files:

- `results/baseline_mos_raw.csv`
- `results/baseline_mos_summary.csv`

## Setup on RunPod

```bash
git clone https://github.com/Shrivar3/twi-tone-conditioned-vits.git
cd twi-tone-conditioned-vits

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
hf auth login
```

## Step 1: inspect Farmerline baseline model files

```bash
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("FarmerlineML/twi-tts-2026", repo_type="model")
for f in files:
    print(f)
PY
```

## Step 2: generate baseline audio

```bash
PYTHONPATH=. python scripts/02_run_baseline_tts.py --config configs/week1_eval.yaml
```

## Step 3: run ASR and WER

```bash
PYTHONPATH=. python scripts/03_run_asr_wer.py --config configs/week1_eval.yaml
```

## Step 4: summarise WER

```bash
PYTHONPATH=. python scripts/07_summarise_wer.py   --input results/baseline_wer.csv   --output results/baseline_wer_summary.csv
```

## Notes

The baseline TTS loader may need adjustment depending on the exact structure of `FarmerlineML/twi-tts-2026`. Inspect the model files before debugging the synthesis script.
