# Week 2 improvement plan: model inspection, data audit, and tone-conditioning preparation

This plan assumes the current Farmerline instruction: **Gemini tone annotations may be treated as the working reference labels for now**, with native-speaker review used later as a validation/audit step rather than as a blocker.

## What changes from Week 1

In Week 1, Gemini tone labels were treated as candidate labels requiring native review before use. Under the current Farmerline instruction, we can now use Gemini labels as provisional ground truth for:

- building a tone-labelled development set;
- computing tone-token distributions;
- preparing minimal-pair prompts;
- designing a tone-conditioned training manifest;
- running early ablation experiments once GPU access is available.

The important caveat is that the labels should still preserve provenance fields such as `gemini_model`, `gemini_sentence_confidence`, `gemini_token_confidences`, `gemini_needs_native_review`, and `gemini_items_json`. That lets us later compare Gemini labels against native-speaker corrections without losing auditability.

## Immediate no-GPU work

### 1. Confirm the current model repository

The repo previously referenced `FarmerlineML/twi-tts-2026`, while the latest shared model is `FarmerlineML/main_twi_TTS`. Before running synthesis, inspect the model repository files:

```bash
PYTHONPATH=. python scripts/10_inspect_baseline_model.py \
  --model-id FarmerlineML/main_twi_TTS \
  --output results/model_file_inventory.csv
```

Use the output to check:

- whether the model is a Hugging Face `transformers` VITS checkpoint;
- whether the checkpoint has `config.json`, `vocab.json`, tokenizer files, or custom code;
- expected sample rate;
- tokenizer vocabulary and special tokens;
- whether `src/eval/synth_baseline.py` can load it directly or needs a custom loader.

### 2. Audit the Farmerline TTS dataset

```bash
PYTHONPATH=. python scripts/11_audit_farmerline_dataset.py \
  --split train \
  --streaming \
  --summary-output results/farmerline_dataset_audit_summary.csv
```

Key questions:

- Are there unexpected characters that the model tokenizer does not cover?
- Are there duplicate texts or possible train/test leakage?
- Are duration and text length consistent?
- Are there malformed rows, empty transcripts, or suspiciously fast/slow text-duration pairs?

### 3. Audit the Asante Twi phoneme dataset

```bash
PYTHONPATH=. python scripts/12_audit_phoneme_dataset.py \
  --split train \
  --streaming \
  --summary-output results/phoneme_dataset_audit_summary.csv
```

The phoneme dataset should be treated as a pronunciation/G2P resource first, not as direct lexical-tone supervision. It can still help us prepare phoneme-conditioned experiments.

### 4. Build a phoneme inventory

```bash
PYTHONPATH=. python scripts/13_build_phoneme_inventory.py \
  --splits train validation test \
  --output results/phoneme_inventory.csv
```

This tells us the set of phoneme symbols that a phoneme-conditioned model would need to support.

### 5. Compare Farmerline text against the phoneme dataset transcripts

```bash
PYTHONPATH=. python scripts/14_compare_farmerline_to_phoneme_vocab.py \
  --farmerline-split train \
  --phoneme-split train \
  --summary-output results/farmerline_vs_phoneme_vocab_summary.csv
```

This helps decide whether the phoneme dataset is close enough in text/domain coverage to help the Farmerline TTS model.

## Recommended GPU experiments once access is ready

Run these in order, so we can attribute improvements cleanly.

### Experiment 0: current model baseline

- Input: original Twi text.
- Output: generated audio.
- Evaluation: MOS template, ASR/WER, and minimal-pair tone accuracy.
- Purpose: establish the current checkpoint quality before changing anything.

### Experiment 1: text normalisation only

- Input: normalised Twi text.
- No model architecture change.
- Purpose: test whether unicode, punctuation, apostrophes, or character normalisation improve stability.

### Experiment 2: Gemini tone-tagged text

- Input: text with inline tone tags, for example `asɛm<L-H>` or a safer separated-token representation.
- Purpose: test whether the model can learn from tone information without modifying the VITS architecture.

### Experiment 3: phoneme input or mixed grapheme+phoneme input

- Input: phoneme sequence or mixed text/phoneme representation.
- Purpose: improve pronunciation and reduce tokenizer ambiguity.

### Experiment 4: explicit tone embedding branch

- Input: text tokens plus separate tone-token sequence.
- Requires model-code changes.
- Purpose: cleaner tone conditioning than inline tags.

## Suggested success criteria

- Baseline synthesis runs end-to-end on the fixed development set.
- Character/token inventory has no unexplained unsupported symbols.
- Gemini-labelled tone manifest is generated and committed as a small CSV manifest.
- Minimal-pair tone evaluation can be run before and after any model change.
- Every model improvement is compared against the same fixed dev/minimal-pair set.

## Practical notes

- Do not commit raw audio, generated audio, checkpoints, Hugging Face cache files, or API keys.
- Commit small manifests, CSV summaries, documentation, and scripts only.
- Keep Gemini output metadata even if treating the labels as correct for now.
- Native-speaker review should later be used to estimate Gemini label error, not to block current engineering work.
