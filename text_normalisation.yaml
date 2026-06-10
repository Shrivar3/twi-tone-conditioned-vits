# Twi Tone-Conditioned VITS

Starter repository for the RAIC / Farmerline Week 1 work on Twi speech synthesis.

## Week 1 goal

1. Load and audit `FarmerlineML/Twi_TTS2026_dataset` from Hugging Face.
2. Create a fixed development set for comparable evaluation.
3. Run baseline TTS using the current Farmerline checkpoint.
4. Run round-trip ASR and compute WER/CER.
5. Create a conservative Akan/Twi tone annotation pipeline.
6. Produce a tone-annotated development set and native-speaker validation sheet.

## Repository rules

Keep GitHub for code, configs, small CSV manifests, and result summaries only.
Do **not** commit raw datasets, generated audio, model checkpoints, Hugging Face cache files, or secrets.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
hf auth login
```

## Suggested Week 1 workflow

```bash
# 1. Create a fixed dev set
python scripts/01_make_dev_set.py --config configs/week1_eval.yaml

# 2. Run baseline TTS on MOS/WER subset
python scripts/02_run_baseline_tts.py --config configs/week1_eval.yaml

# 3. Run ASR and compute WER/CER
python scripts/03_run_asr_wer.py --config configs/week1_eval.yaml

# 4. Create tone-annotated dev set and validation sheet
python scripts/04_annotate_tone.py --config configs/week1_eval.yaml
```

## Outputs

Expected small files to commit:

- `data/manifests/dev_set.csv`
- `data/manifests/tone_annotated_dev.csv`
- `data/manifests/native_validation_sheet.csv`
- `results/baseline_wer.csv`
- `results/baseline_mos_template.csv`
- `results/week1_summary.md`

Generated audio should remain local under `outputs/` and should not be committed.

## Notes

The baseline synthesis code in `src/eval/synth_baseline.py` uses a generic Transformers-style VITS loader. If the Farmerline checkpoint uses custom code or a different TTS framework, only the `BaselineSynthesiser.synthesise_one` method should need adapting.
