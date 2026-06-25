# Week 1 Summary

Updated: 2026-06-25

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
- GPU status: baseline TTS regenerated on the GPU instance (2026-06-25)
- Generated samples: 50 WER dev-set samples
- Output handling: generated audio kept outside GitHub; only small manifests/results committed

## Round-trip ASR/WER

Current status: **valid (corrected run, 2026-06-25)**

The ASR/WER step has been rerun with the correct Twi ASR model and now produces a valid baseline. The earlier WER/CER = 1.0 was a failed Whisper diagnostic run and has been replaced.

What was fixed:

- The config now uses `FarmerlineML/twi-asr-qwen2audio-merged` (backend: qwen2_audio), not Whisper.
- The previous Whisper setup failed with `Unsupported language: twi`, leaving empty transcripts and defaulting WER/CER to 1.0.
- The corrected run produced non-empty transcripts for all 50 samples.

Run summary:

- Number of samples: 50
- Rows with ASR errors: 0
- Rows with empty transcripts: 0

Valid baseline WER/CER:

- Mean WER: 0.906
- Median WER: 0.875
- Mean CER: 0.447
- Median CER: 0.381

Interpretation: these numbers are high, as expected for the un-modified baseline model. They are the valid reference point for measuring improvement after tone conditioning.

Output files (pushed to GitHub):

- `results/baseline_wer.csv`
- `results/baseline_wer_summary.csv`
- `results/asr_transcripts.csv`

## MOS

Current status: **not yet measured**

Pending:

- Baseline MOS naturalness
- Baseline MOS intelligibility
- Native-rater evaluation workflow
- Farmerline coordination for native raters

(Likely a Week 3 activity per the 4-week plan. Brief reference baseline MOS: 3.5/5.0.)

## Tone annotation

- Gemini tone outputs are currently treated as the working reference labels, following Farmerline guidance.
- Native-speaker review remains important but is currently a later audit/correction step rather than a blocker.
- Native validation materials have been prepared and sent to a native reviewer (Akosua); review return expected early next week.

## No-GPU audit and Week 2 preparation

Completed no-GPU preparation includes:

- Farmerline dataset audit
- Asante Twi phoneme dataset audit
- Phoneme inventory generation (full 1000-row run, replacing the earlier 200-row sample)
- Farmerline-vs-phoneme vocabulary comparison (full 1000-row run)
- Hugging Face model repository inspection

Key interpretation:

The Asante Twi phoneme dataset is useful as a possible phoneme-informed resource, but it differs substantially from the Farmerline dataset in domain, casing, punctuation, Unicode conventions, and token distribution. Normalisation and careful alignment will be needed before using it directly for modelling.

## Pending work

Completed since last update:

1. ASR code/config switched to `FarmerlineML/twi-asr-qwen2audio-merged`.
2. New GPU instance obtained and used.
3. Baseline TTS audio regenerated (50 samples).
4. ASR rerun on all 50 baseline samples (0 errors, 0 empty transcripts).
5. Valid WER/CER computed and this summary updated.
6. Full phoneme inventory + vocab comparison rerun and pushed.

Still pending:

- Baseline MOS naturalness and intelligibility (needs native raters; likely Week 3)
- Minimal-pair tone-accuracy evaluation (needs a reviewed minimal-pair set + native raters)
- Native-speaker audit of Gemini tone labels (in progress with Akosua)
- Final decision on tone-conditioning format (syllable-level vs token-level) with Farmerline
- Longer/stable GPU access for the Week 2 training + ablations

## Week 1 conclusion

Week 1 preparation and baseline evaluation are now complete on the GPU side: baseline TTS audio, a valid round-trip WER/CER, and the full phoneme audit have all been produced and pushed.

Remaining items (native validation, MOS, minimal-pair) do not require GPU and are either in progress or planned for later weeks. The main thing to secure before Week 2 is longer/stable GPU access for the tone-conditioned training and ablations.
