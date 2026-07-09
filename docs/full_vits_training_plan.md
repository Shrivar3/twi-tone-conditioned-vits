# Full VITS Tone Training Plan

## Why the proxy loop was insufficient

`scripts/17_train_tone_conditioned_vits.py` was useful as a gradient-flow and
adapter smoke test, but it trained against waveform MSE after running the
inference path. That is not the VITS objective. It does not train a posterior
encoder, does not learn monotonic text-to-acoustic alignment, does not optimise
the normalising flow, does not use mel reconstruction, and does not use an
adversarial discriminator. It can show that tone embeddings are wired in, but it
cannot produce a production-quality VITS fine-tune.

## What full VITS training adds

The new `src/training/full_vits/` package follows the original VITS training
structure:

- tone-aware text prior encoder;
- posterior encoder over linear spectrograms;
- residual coupling flow between posterior and prior latents;
- monotonic alignment search for text-to-spectrogram alignment;
- duration predictor and duration loss;
- HiFi-GAN-style waveform decoder;
- multi-scale/multi-period discriminator;
- mel reconstruction loss;
- KL loss;
- adversarial discriminator and generator losses;
- feature matching loss;
- checkpoint save/resume and validation synthesis samples.

Tone conditioning is controlled by `tone_conditioning_mode`:

- `none`: disable tone conditioning for ablations;
- `additive`: add tone embeddings to text embeddings;
- `concat_projection`: concatenate text and tone embeddings, then project back
  to the hidden size. This is the default.

## Prepare filelists

Create VITS-style filelists from a Farmerline manifest with audio paths:

```bash
python scripts/21_prepare_full_vits_filelists.py \
  --manifest data/manifests/farmerline_train_with_tones.csv \
  --audio-column audio_path \
  --output-dir data/filelists/full_vits \
  --seed 42
```

Each line is:

```text
audio_path|text|gemini_tone_sequence
```

The training dataset tokenises Twi text with the Farmerline tokenizer and uses
`gemini_tone_sequence` to build token-aligned `tone_ids`.

## Debug run

Use the small debug config first:

```bash
python scripts/22_train_full_tone_vits.py \
  --config configs/full_vits_tone_debug.yaml \
  --device cuda
```

Outputs go under `checkpoints/full_vits_tone_debug/`, which is gitignored. The
debug config uses smaller channels and frequent validation/checkpoint intervals.

For a no-tone ablation with the same full objective:

```bash
python scripts/22_train_full_tone_vits.py \
  --config configs/full_vits_tone_debug.yaml \
  --disable-tone-conditioning
```

## Real GPU training run

After the debug run completes a few checkpoints and validation samples:

```bash
python scripts/22_train_full_tone_vits.py \
  --config configs/full_vits_tone_train.yaml \
  --device cuda
```

Resume from a saved checkpoint:

```bash
python scripts/22_train_full_tone_vits.py \
  --config configs/full_vits_tone_train.yaml \
  --resume-from-checkpoint checkpoints/full_vits_tone_train/full_tone_vits_step_00050000.pt \
  --device cuda
```

Do not commit generated WAV files, checkpoints, caches, or local datasets.

## Full-tone synthesis

Generate a quick two-sample manifest from a saved full-VITS checkpoint:

```bash
python scripts/23_synth_full_tone_vits.py \
  --checkpoint checkpoints/full_vits_tone_debug/full_tone_vits_final.pt \
  --manifest data/manifests/dev_set_tone_conditioning_gpu_check.csv \
  --output-dir outputs/full_tone_vits_debug \
  --output-manifest data/manifests/dev_set_with_full_tone_vits_audio.csv \
  --max-samples 2
```

The output manifest keeps all source columns and adds
`full_tone_vits_audio_path`. Use `--disable-tone-conditioning` with the same
checkpoint for a no-tone ablation synthesis pass.

## Smoke tests

The smoke tests cover:

- one dataloader batch;
- one generator/discriminator forward pass;
- one tiny training step.

Run them with:

```bash
python -m unittest tests.test_full_vits_smoke
```

## Evaluation

Use the existing baseline and tone-conditioned evaluation flow:

1. Synthesize baseline audio with `scripts/02_run_baseline_tts.py`.
2. Synthesize full-tone model outputs from validation/test text with
   `scripts/23_synth_full_tone_vits.py`.
3. Run ASR WER/CER on baseline and full-tone outputs with
   `scripts/03_run_asr_wer.py`.
4. Build blind MOS sheets with `scripts/19_make_blind_mos_comparison_sheet.py`.
5. Summarise MOS and WER with the existing summary scripts.
6. Use the Wilcoxon comparison CSV workflow already in `results/` to compare
   paired baseline vs full-tone WER/CER, and report MOS separately as blind
   rater scores.

Primary success criteria should include native tone accuracy/MOS improvements
without a WER regression. WER alone is not enough because tone improvements can
be perceptually meaningful even when ASR text is unchanged.
