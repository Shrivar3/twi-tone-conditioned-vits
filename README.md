# Twi Tone-Conditioned VITS

Repository for the RAIC / Farmerline Group Ghana Twi speech synthesis project.

The project investigates tone-aware text-to-speech for Twi/Akan, starting from the current Farmerline TTS data and baseline model, then moving toward tone-conditioned or phoneme-informed VITS-style improvements.

## Current project status

The repository has moved beyond the initial no-GPU preparation stage.

A first GPU/RunPod baseline run has been completed. Baseline TTS audio was generated for 50 WER development-set samples using the Farmerline baseline TTS model currently recorded in the config as `FarmerlineML/main_twi_TTS`.

The main unresolved issue is round-trip ASR/WER. The committed ASR/WER outputs are invalid because the ASR step used a Whisper-style configuration with `language: twi`, which failed with `Unsupported language: twi`. The resulting WER/CER values of 1.0 are diagnostic failure outputs, not real baseline scores.

The next GPU session should focus on rerunning ASR/WER with the correct Farmerline ASR model, `FarmerlineML/twi-asr-qwen2audio-merged`.

Current useful work before the next GPU instance:

- update repo status and summary documentation;
- patch the ASR code/config for Qwen2-Audio inference;
- prepare a one-sample ASR smoke test;
- keep generated audio/checkpoints out of GitHub;
- confirm how generated audio and larger result artefacts should be stored.

## Current tone-label assumption

Farmerline has instructed us to treat Gemini tone outputs as the working reference labels for current development.

Native-speaker review remains important and will be used later as an audit/correction layer, but it is not currently a blocker for using Gemini tone labels in early manifests, evaluation design, or tone-conditioning preparation.

## Current access status

The following datasets are accessible:

- `FarmerlineML/Twi_TTS2026_dataset`
- `ghananlpcommunity/asante-twi-bible-speech-phonemes`

The baseline TTS model currently recorded in the config is:

- `FarmerlineML/main_twi_TTS`

The ASR model that should be used for the corrected WER run is:

- `FarmerlineML/twi-asr-qwen2audio-merged`

Generated audio, model checkpoints, Hugging Face caches, local RunPod outputs, and API keys should not be committed to GitHub.

## Completed / prepared

* GitHub repository structure created.
* Hugging Face dataset access tested.
* Codespaces-safe development set utilities added.
* Conservative Twi/Akan tone annotation pipeline added.
* Gemini tone annotation pipeline added.
* Native-speaker token-level validation workflow added.
* MOS, WER, and minimal-pair tone accuracy templates added.
* Summary scripts for MOS, WER, and native-validation outputs added.
* No-GPU Week 2 inspection utilities added:

  * Farmerline dataset audit.
  * Asante Twi phoneme dataset audit.
  * Phoneme inventory generation.
  * Farmerline-vs-phoneme vocabulary comparison.
  * Hugging Face model repository inspection.

## Important repository rules

GitHub should contain code, configs, small CSV manifests, templates, and result summaries only.

Do **not** commit:

* raw datasets;
* generated audio;
* model checkpoints;
* Hugging Face cache files;
* API keys or secrets;
* local virtual environments;
* local patch bundles;
* backup files.

## Repository structure

```text
configs/              YAML configs for evaluation and inspection
src/data/             Dataset loading and development-set utilities
src/tone/             Tone annotation and validation utilities
src/eval/             Baseline synthesis, ASR, MOS, and WER utilities
scripts/              Command-line entry points
data/manifests/       Small CSV manifests and validation sheets
data/minimal_pairs/   Minimal-pair templates for tone accuracy testing
docs/                 Protocols, plans, and handoff notes
results/              Result templates, summaries, and small audit outputs
outputs/              Local generated outputs; ignored by Git
```

## Light Codespaces setup

Use this environment for editing code, creating small manifests, Gemini annotation preparation, validation sheets, and no-GPU data audits.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gemini.txt
python -m pip install datasets pandas pyyaml huggingface_hub
```

Avoid committing local environments, generated audio, large datasets, model checkpoints, or local patch bundles.

## Week 2 no-GPU inspection workflow

### Farmerline dataset audit

```bash
PYTHONPATH=. python scripts/11_audit_farmerline_dataset.py \
  --split train \
  --streaming \
  --limit 1000
```

Expected small outputs:

```text
results/farmerline_dataset_audit_summary.csv
results/farmerline_character_inventory.csv
results/farmerline_token_inventory.csv
results/farmerline_text_duration_outliers.csv
```

### Asante Twi phoneme dataset audit

```bash
PYTHONPATH=. python scripts/12_audit_phoneme_dataset.py \
  --split train \
  --streaming \
  --limit 1000
```

Expected small outputs:

```text
results/phoneme_dataset_audit_summary.csv
results/phoneme_inventory.csv
results/phoneme_speaker_summary.csv
results/phoneme_long_clip_candidates.csv
```

### Build phoneme inventory

```bash
PYTHONPATH=. python scripts/13_build_phoneme_inventory.py \
  --limit-per-split 1000
```

Expected output:

```text
results/phoneme_inventory.csv
```

### Compare Farmerline and phoneme-dataset vocabulary

```bash
PYTHONPATH=. python scripts/14_compare_farmerline_to_phoneme_vocab.py \
  --farmerline-limit 1000 \
  --phoneme-limit 1000
```

Expected outputs:

```text
results/farmerline_vs_phoneme_character_overlap.csv
results/farmerline_vs_phoneme_token_overlap.csv
results/farmerline_vs_phoneme_vocab_summary.csv
```

### Inspect baseline model repository access

```bash
PYTHONPATH=. python scripts/10_inspect_baseline_model.py \
  --model-id FarmerlineML/main_twi_TTS \
  --output results/model_file_inventory.csv
```

Current result: model repository access is not yet working for the tested model IDs, even though dataset access works.

## Initial no-GPU audit findings

Farmerline sample audit, first 1,000 train rows:

* Empty text rows: 0
* Duplicate text rows: 0
* Median duration: 6.44 seconds
* Duration range: 2.00 to 11.72 seconds
* Unique characters: 31
* Unique lowercase tokens: 2,969

Asante Twi phoneme dataset sample audit, first 1,000 train rows:

* Empty text rows: 0
* Empty phoneme rows: 0
* Unique phoneme tokens: 35
* Unique speakers: 10

Farmerline-vs-phoneme comparison, first 1,000 rows each:

* Farmerline unique characters: 31
* Phoneme dataset unique characters: 52
* Farmerline unique tokens: 2,969
* Phoneme dataset unique tokens: 4,275
* Farmerline-only token count: 2,093
* Phoneme-dataset-only token count: 3,399

Interpretation: the phoneme dataset is useful, but domain and text-normalisation differences are substantial. Case, punctuation, Unicode handling, and tokenisation should be normalised before using it directly for phoneme-informed modelling.

## Current blockers

* GPU compute access is pending.
* Higher Gemini quota/access is pending.
* Farmerline should confirm the exact Hugging Face model repo ID and grant access if the repo is private/gated.
* Farmerline should confirm where generated audio and result artefacts should be stored.
