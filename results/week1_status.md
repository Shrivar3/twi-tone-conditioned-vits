# Project Status

## Completed / prepared

* GitHub repository structure created.
* Hugging Face dataset access tested.
* Codespaces-safe development set utilities added.
* Conservative Twi/Akan tone annotation pipeline added.
* Gemini tone annotation pipeline added.
* Native-speaker token-level validation workflow added.
* MOS, WER, and minimal-pair tone accuracy templates added.
* Summary scripts for MOS, WER, and native-validation outputs added.
* RunPod/GPU baseline evaluation plan added.
* Week 2 no-GPU inspection utilities added:

  * Farmerline dataset audit.
  * Asante Twi phoneme dataset audit.
  * Phoneme inventory generation.
  * Farmerline-vs-phoneme vocabulary comparison.
  * Hugging Face model repository inspection.

## Current tone-label assumption

Farmerline has instructed us to treat Gemini tone outputs as the working reference labels for current development.

Native-speaker review remains planned as a later audit/correction layer, but it is not currently a blocker for using Gemini labels in early manifests, evaluation design, or tone-conditioning preparation.

## Initial no-GPU audit results

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

## Not yet completed

* Baseline Farmerline TTS audio generation.
* Round-trip ASR/WER.
* Baseline MOS naturalness.
* Baseline MOS intelligibility.
* Minimal-pair tone accuracy test.
* Full native-speaker audit of Gemini tone labels.
* Baseline model repository inspection, because the tested model repo IDs are not currently accessible.

## Current measurable outcomes

| Metric                         |   Current status |    Target / success threshold |
| ------------------------------ | ---------------: | ----------------------------: |
| Twi TTS MOS naturalness        | Not yet measured |               Above 3.8 / 5.0 |
| Twi TTS MOS intelligibility    | Not yet measured |               Above 3.8 / 5.0 |
| Round-trip ASR WER             | Not yet measured | Lower after tone conditioning |
| Tone accuracy on minimal pairs | Not yet measured |                     Above 80% |

## Current blockers

* GPU compute access is pending.
* Higher Gemini quota/access is pending.
* Farmerline should confirm the exact Hugging Face model repo ID and grant access if the repo is private/gated.
* Farmerline should confirm where generated audio and result artefacts should be stored.
