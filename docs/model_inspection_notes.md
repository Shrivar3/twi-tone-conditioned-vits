# Model inspection notes

## Current issue

The repo previously referenced `FarmerlineML/twi-tts-2026` as the baseline checkpoint, while the newly shared model is `FarmerlineML/main_twi_TTS`. Before using GPU time, the team should confirm which model is the official baseline.

## Why this matters

The baseline synthesis code assumes the model can be loaded through the Hugging Face `transformers` VITS path. That will only work if the model repository has compatible config/tokenizer/checkpoint files. If the model is copied from a Facebook MMS-TTS style VITS checkpoint, this may work directly. If it uses custom code, a different tokenizer, or missing metadata, the loader needs to be adjusted before running a full evaluation.

## No-GPU inspection command

```bash
PYTHONPATH=. python scripts/10_inspect_baseline_model.py \
  --model-id FarmerlineML/main_twi_TTS \
  --output results/model_file_inventory.csv
```

## What to check in `results/model_file_inventory.csv`

1. `config.json`
   - Does it exist?
   - What is `model_type`?
   - What are the listed `architectures`?
   - Is there a sampling rate?

2. Tokenizer files
   - Does `vocab.json` exist?
   - Are Twi characters such as `ɛ`, `ɔ`, and nasal/diacritic forms represented?
   - Are `[PAD]`, `[UNK]`, and other special tokens defined?

3. Checkpoint files
   - Is the checkpoint `pytorch_model.bin`, `model.safetensors`, or something else?
   - Is the model size consistent with a small VITS checkpoint?

4. Custom code
   - Are there Python files in the model repo?
   - If yes, the loader may need `trust_remote_code=True` or a custom loading path.

## Next decision after inspection

- If `transformers.VitsModel` and `AutoTokenizer` load cleanly, proceed with baseline synthesis.
- If the model files are compatible but the tokenizer misses Twi characters, fix text normalisation/tokenizer handling before synthesis.
- If the model is not directly compatible with `transformers`, write a custom loader before any full GPU run.

## Suggested follow-up command once model loading is confirmed

```bash
PYTHONPATH=. python scripts/02_run_baseline_tts.py --config configs/week1_eval.yaml
```

Then run:

```bash
PYTHONPATH=. python scripts/03_run_asr_wer.py --config configs/week1_eval.yaml
PYTHONPATH=. python scripts/07_summarise_wer.py \
  --input results/baseline_wer.csv \
  --output results/baseline_wer_summary.csv
```
