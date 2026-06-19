# Week 1 Summary

## Dataset
- Dataset: `FarmerlineML/Twi_TTS2026_dataset`
- Split used for dev set: `test` split (dev rows map to test indices 0–51)
- Number of WER samples: 50 (`wer_subset = True`)
- Number of MOS samples: MOS subset flagged in the dev set (`mos_subset = True`) — confirm exact count from `data/manifests/dev_set.csv`
- Number of tone-validation samples: 10 utterances (`dev_0000`–`dev_0009`), ~160 tokens
- Duration filter: dev-set clips ~5–9 s; full-dataset range 2.00–11.72 s (median 6.44 s)

## Baseline TTS
- Checkpoint: status file lists `FarmerlineML/twi-tts-2026`, but Farmerline later confirmed the working repo is `FarmerlineML/main_twi_TTS` — **resolve this discrepancy and record the repo actually used**
- Inference code path: `scripts/02_run_baseline_tts.py` → `src/eval/synth_baseline.py`
- Number of generated samples: 50 (`dev_0000`–`dev_0049`)
- Notes/errors: ran successfully on GPU (RunPod); audio generated for all 50 samples

## Round-trip ASR WER
- ASR model: ⚠️ a **Whisper-type** model was used (the error lists Whisper's supported languages). This is **incorrect** — the brief requires `FarmerlineML/twi-asr-qwen2audio-merged`
- Text normalisation: n/a (no transcripts were produced)
- Mean WER: 1.0 — **INVALID**
- Median WER: 1.0 — **INVALID**
- Mean CER: 1.0 — **INVALID**
- Median CER: 1.0 — **INVALID**
- Caveats: every one of the 50 rows failed with `ValueError: Unsupported language: twi`, so the `asr_transcript` column is empty and WER/CER defaulted to 1.0. **These are not real scores.** The WER step must be re-run on GPU using the correct Twi ASR model.

## MOS
- Number of raters: not yet run (Farmerline to coordinate 10–20 native raters)
- Number of samples: target 50 per rater
- Naturalness MOS: not yet measured (baseline reference from brief: 3.5/5.0)
- Intelligibility MOS: not yet measured
- Caveats: requires the MOS evaluation interface + Farmerline-coordinated raters

## Tone annotation
- Automatic method: Gemini (`gemini-2.5-flash`)
- Number of tokens labelled H: compute from `data/manifests/gemini_tone_annotated_dev.csv`
- Number of tokens labelled L: compute from full file (note: low tone is heavily dominant in the sample)
- Number of tokens labelled F: compute from full file
- Number of tokens labelled UNK: compute from full file (mostly loanwords/proper nouns, e.g. "minnesota")
- Native validation subset size: 10 utterances (~160 tokens) — sheet prepared, sent to native reviewer (Akosua)
- Native validation accuracy: pending reviewer return
- Main issues discovered: Gemini over-assigns L (low tone); loanwords and proper nouns get UNK or low confidence; syllable-level vs token-level format still to be decided with Farmerline

## Week 2 readiness
- Is the baseline pipeline reproducible? Partially. Baseline TTS generation works; the ASR/WER step is broken (wrong ASR model) and must be fixed before the baseline is valid.
- Are tone labels credible enough for model conditioning? Farmerline has approved Gemini outputs as working reference labels; native audit (in progress) will confirm quality.
- What needs changing before modifying VITS:
  1. Fix the ASR/WER step (use `FarmerlineML/twi-asr-qwen2audio-merged`) and produce a valid baseline WER.
  2. Complete the native tone audit on the pilot sheet.
  3. Confirm the baseline model repo ID actually used.
  4. Agree syllable-level vs token-level tone format with Farmerline.
