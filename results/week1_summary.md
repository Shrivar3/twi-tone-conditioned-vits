# Week 1 Summary

Updated: 2026-06-24

## Dataset

- Dataset: `FarmerlineML/Twi_TTS2026_dataset`
- Split used for dev set: `test`
- WER subset: 50 samples, `dev_0000` to `dev_0049`
- MOS subset: flagged in `data/manifests/dev_set.csv`; intended sample count from config is 20
- Tone-validation subset: prepared separately for native validation
- Duration filter used for dev-set preparation: approximately 2-10 seconds
- Initial audit result: full sampled Farmerline duration range was approximately 2.00-11.72 seconds, with median duration 6.44 seconds

## Baseline TTS

- Baseline TTS model currently recorded in config: `FarmerlineML/main_twi_TTS`
- Inference code path: `scripts/02_run_baseline_tts.py` to `src/eval/synth_baseline.py`
- GPU status: baseline TTS was run once on GPU/RunPod
- Generated samples: 50 WER dev-set samples
- Output handling: generated audio should remain outside GitHub; only small manifests/results should be committed

## Round-trip ASR/WER

Current status: **invalid / must be rerun**

The ASR/WER run currently committed to the repo should be treated as a failed diagnostic run, not as a real baseline result.

What happened:

- The current config uses `openai/whisper-small` with `language: twi`.
- This is the wrong ASR setup for the project brief.
- The ASR run failed with `Unsupported language: twi`.
- The ASR transcript column is empty for the failed rows.
- WER/CER values of 1.0 were produced because the hypotheses were empty after ASR failure.

Current WER/CER values:

- Mean WER: 1.0 — **invalid**
- Median WER: 1.0 — **invalid**
- Mean CER: 1.0 — **invalid**
- Median CER: 1.0 — **invalid**

These values must not be reported as baseline performance.

Required correction:

- Use `FarmerlineML/twi-asr-qwen2audio-merged`
- Do not use Whisper language forcing for Twi
- Run a one-sample smoke test first
- Then rerun ASR over all 50 generated baseline samples
- Recompute WER/CER only after non-empty ASR transcripts are produced

## MOS

Current status: **not yet measured**

Pending:

- Baseline MOS naturalness
- Baseline MOS intelligibility
- Native-rater evaluation workflow
- Farmerline coordination for native raters

## Tone annotation

- Gemini tone outputs are currently treated as the working reference labels, following Farmerline guidance.
- Native-speaker review remains important but is currently a later audit/correction step rather than a blocker.
- Native validation materials have been prepared, but full native-speaker validation is still pending.

## No-GPU audit and Week 2 preparation

Completed no-GPU preparation includes:

- Farmerline dataset audit
- Asante Twi phoneme dataset audit
- Phoneme inventory generation
- Farmerline-vs-phoneme vocabulary comparison
- Hugging Face model repository inspection

Key interpretation:

The Asante Twi phoneme dataset is useful as a possible phoneme-informed resource, but it differs substantially from the Farmerline dataset in domain, casing, punctuation, Unicode conventions, and token distribution. Normalisation and careful alignment will be needed before using it directly for modelling.

## Pending work

Highest-priority pending work:

1. Prepare ASR code/config for `FarmerlineML/twi-asr-qwen2audio-merged`.
2. Obtain a new GPU instance.
3. Recreate or verify baseline TTS audio files.
4. Run one-sample Qwen2-Audio ASR smoke test.
5. Rerun ASR on all 50 baseline generated samples.
6. Recompute real WER/CER.
7. Update this summary with valid WER/CER numbers.

Still pending after ASR/WER:

- Baseline MOS naturalness
- Baseline MOS intelligibility
- Minimal-pair tone-accuracy evaluation
- Native-speaker audit of Gemini tone labels
- Final decision on tone-conditioning format for model modification

## Week 1 conclusion

Week 1 preparation and the first baseline TTS generation step are substantially complete.

The baseline evaluation is not complete because WER/CER is currently invalid. The next GPU session should focus on fixing ASR/WER before reporting any quantitative baseline speech-recognition metrics.
