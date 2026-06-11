# Week 1 Status

## Completed

- GitHub repository structure created.
- Hugging Face access confirmed.
- Codespaces-safe dev set created: `data/manifests/dev_set.csv`.
- Conservative tone annotation created: `data/manifests/tone_annotated_dev.csv`.
- Native validation sheet created: `data/manifests/native_validation_sheet.csv`.
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
