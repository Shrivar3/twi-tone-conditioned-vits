# Project Status

Updated: 2026-06-24

## Overall status

The repository has moved beyond the initial no-GPU preparation stage.

A first GPU/RunPod baseline run has been completed. Baseline TTS audio was generated for the 50 WER development-set samples using the Farmerline baseline TTS model currently recorded in the config as `FarmerlineML/main_twi_TTS`.

However, the round-trip ASR/WER result is not valid yet. The ASR step used a Whisper-style model/configuration, which failed for Twi with `Unsupported language: twi`. As a result, the committed WER/CER values are diagnostic failure outputs only and must not be treated as real baseline scores.

The next GPU session should prioritise fixing the ASR path and rerunning WER using `FarmerlineML/twi-asr-qwen2audio-merged`.

## Completed / prepared

- GitHub repository structure created.
- Hugging Face dataset access tested.
- Codespaces-safe development-set utilities added.
- Conservative Twi/Akan tone annotation pipeline added.
- Gemini tone annotation pipeline added.
- Native-speaker token-level validation workflow added.
- MOS, WER, and minimal-pair tone-accuracy templates added.
- Summary scripts for MOS, WER, and native-validation outputs added.
- RunPod/GPU baseline evaluation plan added.
- Week 2 no-GPU inspection utilities added:
  - Farmerline dataset audit.
  - Asante Twi phoneme dataset audit.
  - Phoneme inventory generation.
  - Farmerline-vs-phoneme vocabulary comparison.
  - Hugging Face model repository inspection.
- Baseline TTS generation completed once on GPU for 50 dev-set WER samples.
- Baseline audio manifest/results were committed as small CSV artefacts.
- ASR/WER failure mode diagnosed: the current Whisper-style ASR configuration is incompatible with Twi and should be replaced with the Farmerline Qwen2-Audio ASR model.

## Current tone-label assumption

Farmerline has instructed us to treat Gemini tone outputs as the working reference labels for current development.

Native-speaker review remains planned as a later audit/correction layer, but it is not currently a blocker for using Gemini labels in early manifests, evaluation design, or tone-conditioning preparation.

## Initial no-GPU audit results

Farmerline sample audit, first 1,000 train rows:

- Empty text rows: 0
- Duplicate text rows: 0
- Median duration: 6.44 seconds
- Duration range: 2.00 to 11.72 seconds
- Unique characters: 31
- Unique lowercase tokens: 2,969

Asante Twi phoneme dataset sample audit, first 1,000 train rows:

- Empty text rows: 0
- Empty phoneme rows: 0
- Unique phoneme tokens: 35
- Unique speakers: 10

Farmerline-vs-phoneme comparison, first 1,000 rows each:

- Farmerline unique characters: 31
- Phoneme dataset unique characters: 52
- Farmerline unique tokens: 2,969
- Phoneme dataset unique tokens: 4,275
- Farmerline-only token count: 2,093
- Phoneme-dataset-only token count: 3,399

Interpretation: the phoneme dataset is useful, but domain and text-normalisation differences are substantial. Case, punctuation, Unicode handling, and tokenisation should be normalised before using it directly for phoneme-informed modelling.

## Current measurable outcomes

| Metric | Current status | Notes |
|---|---:|---|
| Baseline TTS generation | Completed once | 50 WER dev-set samples generated on GPU. |
| Round-trip ASR/WER | Invalid / not yet measured | Current committed WER/CER values came from failed Whisper-style ASR runs and must not be reported as real scores. |
| Twi TTS MOS naturalness | Not yet measured | Requires native-rater evaluation. |
| Twi TTS MOS intelligibility | Not yet measured | Requires native-rater evaluation. |
| Tone accuracy on minimal pairs | Not yet measured | Requires minimal-pair synthesis and native/automatic evaluation. |
| Gemini/native tone-label agreement | Not yet measured | Native review sheet prepared, but full audit/correction is still pending. |

## Current blockers / risks

- No active GPU instance is available at the moment.
- The ASR/WER code path needs to be updated to use `FarmerlineML/twi-asr-qwen2audio-merged`, not Whisper.
- The Farmerline Qwen2-Audio ASR model may require substantial VRAM and a compatible Transformers installation.
- If the previous RunPod disk was not preserved, baseline TTS audio will need to be regenerated before ASR/WER can be rerun.
- Native-speaker MOS, intelligibility, minimal-pair tone evaluation, and Gemini tone-label audit remain pending.
- Farmerline should confirm where generated audio and larger result artefacts should be stored, since generated audio and checkpoints should not be committed to GitHub.

## Next GPU-session priority order

1. Confirm Hugging Face login/access on the GPU instance.
2. Install a Transformers version that supports Qwen2-Audio.
3. Recreate or verify the baseline audio files referenced by `data/manifests/dev_set_with_baseline_audio.csv`.
4. Replace the Whisper ASR path with `FarmerlineML/twi-asr-qwen2audio-merged`.
5. Run ASR on a single generated sample first and manually inspect the transcript.
6. If the single-sample test is sensible, run ASR over all 50 WER samples.
7. Recompute WER/CER and save a valid `results/baseline_wer.csv`.
8. Only after valid ASR transcripts exist, create/update `results/baseline_wer_summary.csv`.
9. Update `results/week1_summary.md` with the real WER/CER numbers.
