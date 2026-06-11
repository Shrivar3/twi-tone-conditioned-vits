# Twi Tone-Conditioned VITS

Repository for the RAIC / Farmerline Group Ghana Twi speech synthesis project.

The project investigates tone-conditioned VITS-style text-to-speech for Twi/Akan. The immediate Week 1 goal is to establish a clean baseline evaluation workflow, prepare a tone-annotated development set, and set up native-speaker validation before moving into model changes.

## Current Week 1 status

Completed or prepared:

- Fixed development set from `FarmerlineML/Twi_TTS2026_dataset`.
- Conservative Twi/Akan tone annotation pipeline.
- Gemini-assisted candidate tone annotation pipeline for small pilot batches.
- Native-speaker token-level validation workflow.
- MOS, WER, and minimal-pair tone accuracy templates.
- Summary scripts for MOS, WER, and native-validation outputs.
- Baseline evaluation plan for the current `FarmerlineML/twi-tts-2026` checkpoint.

Current blockers / access needed:

- Full baseline TTS and ASR/WER evaluation should be run on GPU compute, not GitHub Codespaces.
- Gemini free-tier quota is too small for large-scale tone annotation.
- Native-speaker review is needed before candidate tone labels can be treated as reference labels.

## Important repository rules

GitHub should contain code, configs, small CSV manifests, templates, and result summaries only.

Do **not** commit:

- raw datasets;
- generated audio;
- model checkpoints;
- Hugging Face cache files;
- API keys or secrets;
- local virtual environments.

## Repository structure

```text
configs/                 YAML configs for Week 1 evaluation
src/data/                Dataset loading and development-set utilities
src/tone/                Tone annotation and validation utilities
src/eval/                Baseline synthesis, ASR, MOS, and WER utilities
scripts/                 Command-line entry points
data/manifests/          Small CSV manifests and validation sheets
data/minimal_pairs/      Minimal-pair templates for tone accuracy testing
docs/                    Protocols and handoff notes
results/                 Result templates and summaries
outputs/                 Local generated outputs; ignored by Git
```

## Light Codespaces setup

Use this for editing code, creating small manifests, Gemini pilot annotation, and validation sheets.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gemini.txt
python -m pip install datasets pandas pyyaml
```

Avoid installing the full `requirements.txt` in Codespaces unless storage/compute has been confirmed. The full TTS/ASR stack can be large.

## Full GPU setup

Use GPU compute, such as the Farmerline/RAIC RunPod environment, for baseline TTS generation, ASR, WER, and later model training.

```bash
git clone https://github.com/Shrivar3/twi-tone-conditioned-vits.git
cd twi-tone-conditioned-vits
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
hf auth login
```

## Week 1 workflow

### 1. Create the fixed development set

```bash
PYTHONPATH=. python scripts/01_make_dev_set.py --config configs/week1_eval.yaml
```

Expected output:

```text
data/manifests/dev_set.csv
```

### 2. Create conservative tone annotations

```bash
PYTHONPATH=. python scripts/04_annotate_tone.py --config configs/week1_eval.yaml
```

Expected outputs:

```text
data/manifests/tone_annotated_dev.csv
data/manifests/native_validation_sheet.csv
```

### 3. Run Gemini candidate tone annotation for small pilot batches

Set the API key in the terminal only. Do not commit it.

```bash
read -s -p "Paste Gemini API key: " GEMINI_API_KEY
echo
export GEMINI_API_KEY
```

Then run a small batch:

```bash
PYTHONPATH=. python scripts/05_gemini_annotate_tone.py \
  --input data/manifests/dev_set.csv \
  --output data/manifests/gemini_tone_annotated_dev.csv \
  --model gemini-2.5-flash-lite \
  --limit 5 \
  --sleep 10 \
  --resume
```

Gemini outputs are candidate annotations only. They require native-speaker validation.

### 4. Prepare native-speaker validation sheet

```bash
PYTHONPATH=. python scripts/08_prepare_native_validation_pack.py \
  --dev-csv data/manifests/dev_set.csv \
  --gemini-csv data/manifests/gemini_tone_annotated_dev.csv \
  --output data/manifests/native_validation_token_sheet.csv \
  --limit-utterances 10
```

After native review, summarise results with:

```bash
PYTHONPATH=. python scripts/09_summarise_native_validation.py \
  --input data/manifests/native_validation_token_sheet.csv \
  --output results/native_validation_summary.csv
```

### 5. Run baseline TTS and WER on GPU compute

This should be run on RunPod or another GPU environment, not Codespaces.

First inspect the baseline model files:

```bash
PYTHONPATH=. python scripts/10_inspect_baseline_model.py
```

Then run baseline synthesis and WER once the model-loading details are confirmed:

```bash
PYTHONPATH=. python scripts/02_run_baseline_tts.py --config configs/week1_eval.yaml
PYTHONPATH=. python scripts/03_run_asr_wer.py --config configs/week1_eval.yaml
PYTHONPATH=. python scripts/07_summarise_wer.py \
  --input results/baseline_wer.csv \
  --output results/baseline_wer_summary.csv
```

## Expected small outputs to commit

```text
data/manifests/dev_set.csv
data/manifests/tone_annotated_dev.csv
data/manifests/native_validation_sheet.csv
data/manifests/native_validation_token_sheet.csv
results/baseline_wer_summary.csv
results/baseline_mos_summary.csv
results/native_validation_summary.csv
results/week1_status.md
```

Generated audio should remain local or be stored in the agreed Farmerline storage location.

## Current next steps

1. Confirm GPU compute access with Farmerline/RAIC.
2. Confirm Hugging Face/model access and storage for generated audio/results.
3. Validate the pilot tone sheet with native Twi/Akan speakers.
4. Run baseline TTS and round-trip ASR/WER on the fixed development set.
5. Agree on the scalable tone annotation route before model modification/training.
