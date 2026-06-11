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
python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("FarmerlineML/twi-tts-2026", repo_type="model")
for f in files:
    print(f)
PY
PYTHONPATH=. python scripts/02_run_baseline_tts.py --config configs/week1_eval.yaml
PYTHONPATH=. python scripts/03_run_asr_wer.py --config configs/week1_eval.yaml
PYTHONPATH=. python scripts/07_summarise_wer.py \
  --input results/baseline_wer.csv \
  --output results/baseline_wer_summary.csv

## 7. Add a Week 1 status file

```bash
cat > results/week1_status.md <<'EOF'
# Week 1 Status

## Completed

- GitHub repository structure created.
- Hugging Face access confirmed.
- Codespaces-safe dev set created:
  - `data/manifests/dev_set.csv`
- Conservative tone annotation created:
  - `data/manifests/tone_annotated_dev.csv`
- Native validation sheet created:
  - `data/manifests/native_validation_sheet.csv`
- Gemini candidate tone annotation pipeline added.
- MOS evaluation templates added.
- Minimal-pair tone accuracy templates added.
- WER summary templates added.
- RunPod baseline evaluation plan added.

## In Progress

- Gemini candidate tone annotations:
  - limited by Gemini free-tier quota
  - continue in small batches
  - treat outputs as candidate labels, not ground truth

## Not Yet Completed

- Baseline Farmerline TTS audio generation
- Round-trip ASR WER
- Baseline MOS naturalness
- Baseline MOS intelligibility
- Minimal-pair tone accuracy test
- Native speaker validation of tone labels

## Current Measurable Outcomes

| Metric | Baseline | Success threshold |
|---|---:|---:|
| Twi TTS MOS naturalness | Not yet measured | Above 3.8 / 5.0 |
| Twi TTS MOS intelligibility | Not yet measured | Above 3.8 / 5.0 |
| Round-trip ASR WER | Not yet measured | Lower after tone conditioning |
| Tone accuracy on minimal pairs | Not yet measured | Above 80% |
