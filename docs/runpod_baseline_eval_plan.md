# RunPod Baseline Evaluation Plan

## Purpose

Run baseline TTS generation and round-trip ASR/WER for the current Farmerline Twi TTS checkpoint.

## Why this should not run in Codespaces

Codespaces is useful for small manifests, validation sheets, and code edits. It is not suitable for full TTS/ASR inference because the full environment can require large Torch/CUDA packages, model files, audio outputs, and dataset cache space.

The baseline TTS/WER step should be run on Farmerline/RAIC GPU compute, such as the A100/RunPod environment mentioned in the brief, or any equivalent GPU environment Farmerline prefers.

## Access needed from Farmerline/RAIC

Please clarify:

- whether a Farmerline/RAIC RunPod account or pod template already exists;
- which GPU instance should be used;
- whether there is a provided Docker image or environment;
- which Hugging Face credentials/model access should be used;
- where generated audio and result files should be stored;
- whether outputs should be uploaded to shared storage, Hugging Face, Google Drive, or another location.

## Inputs

```text
data/manifests/dev_set.csv
```

Baseline model:

```text
FarmerlineML/twi-tts-2026
```

Dataset:

```text
FarmerlineML/Twi_TTS2026_dataset
```

## Expected outputs

Generated baseline audio:

```text
outputs/baseline_tts/wavs/dev_0000.wav
outputs/baseline_tts/wavs/dev_0001.wav
...
```

Baseline TTS manifest:

```text
outputs/baseline_tts/baseline_tts_manifest.csv
```

Round-trip ASR/WER results:

```text
results/baseline_wer.csv
results/baseline_wer_summary.csv
```

MOS files:

```text
results/baseline_mos_raw.csv
results/baseline_mos_summary.csv
```

## Suggested GPU setup

```bash
git clone https://github.com/Shrivar3/twi-tone-conditioned-vits.git
cd twi-tone-conditioned-vits

python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
hf auth login
```

## Step 1: inspect the baseline model files

```bash
PYTHONPATH=. python scripts/10_inspect_baseline_model.py
```

This tells us whether the checkpoint uses a standard Transformers-style VITS layout or a custom TTS framework.

## Step 2: generate baseline audio

```bash
PYTHONPATH=. python scripts/02_run_baseline_tts.py --config configs/week1_eval.yaml
```

The synthesis script may need adjustment depending on the exact structure of `FarmerlineML/twi-tts-2026`.

## Step 3: run ASR and compute WER/CER

```bash
PYTHONPATH=. python scripts/03_run_asr_wer.py --config configs/week1_eval.yaml
```

## Step 4: summarise WER

```bash
PYTHONPATH=. python scripts/07_summarise_wer.py \
  --input results/baseline_wer.csv \
  --output results/baseline_wer_summary.csv
```

## Notes

- Do not commit generated audio to GitHub.
- Keep large outputs in agreed external storage.
- Commit only small manifests and summary CSV files.
