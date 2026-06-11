# Week 1 Status

## Completed / prepared

- GitHub repository structure created.
- Hugging Face access tested.
- Codespaces-safe development set created:
  - `data/manifests/dev_set.csv`
- Conservative tone annotation pipeline added:
  - `data/manifests/tone_annotated_dev.csv`
- Initial native validation sheet created:
  - `data/manifests/native_validation_sheet.csv`
- Gemini candidate tone annotation pipeline added.
- Native-speaker token-level validation workflow added.
- MOS evaluation templates added.
- Minimal-pair tone accuracy templates added.
- WER summary templates added.
- RunPod baseline evaluation plan added.

## In progress

- Gemini candidate tone annotations:
  - limited by free-tier quota;
  - continue only in small batches unless higher quota is provided;
  - treat outputs as candidate labels, not ground truth.
- Native-speaker validation:
  - pilot validation sheet prepared;
  - awaiting reviewer access / feedback.

## Not yet completed

- Baseline Farmerline TTS audio generation.
- Round-trip ASR/WER.
- Baseline MOS naturalness.
- Baseline MOS intelligibility.
- Minimal-pair tone accuracy test.
- Full native-speaker validation of tone labels.

## Current measurable outcomes

| Metric | Current status | Target / success threshold |
|---|---:|---:|
| Twi TTS MOS naturalness | Not yet measured | Above 3.8 / 5.0 |
| Twi TTS MOS intelligibility | Not yet measured | Above 3.8 / 5.0 |
| Round-trip ASR WER | Not yet measured | Lower after tone conditioning |
| Tone accuracy on minimal pairs | Not yet measured | Above 80% |

## Current blockers

- GPU compute access is needed for baseline TTS/ASR evaluation.
- Gemini free-tier quota is too small for large-scale annotation.
- Native-speaker reviewers are needed to validate tone labels.
- Farmerline should confirm where generated audio and result artefacts should be stored.
