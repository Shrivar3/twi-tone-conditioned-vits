# Farmerline Week 1 Native Validation Checklist

## Goal

Prepare a validated Twi/Akan tone reference set for the Week 1 dev set.

## Checklist

### Repository/data preparation

- [ ] `data/manifests/dev_set.csv` exists.
- [ ] `data/manifests/tone_annotated_dev.csv` exists.
- [ ] `data/manifests/gemini_tone_annotated_dev.csv` exists if Gemini candidate labels have been generated.
- [ ] `data/manifests/native_validation_token_sheet.csv` has been generated.

### Native-speaker review

- [ ] Select 2-3 native Twi/Akan speakers for pilot validation.
- [ ] Give reviewers `docs/native_validation_protocol.md`.
- [ ] Give reviewers `data/manifests/native_validation_token_sheet.csv`.
- [ ] Ask reviewers to fill `native_tone_label`, `native_confidence`, `review_status`, and `reviewer_notes`.

### After review

- [ ] Save completed file as `data/manifests/native_validation_token_sheet_completed.csv`.
- [ ] Run `scripts/09_summarise_native_validation.py`.
- [ ] Check `results/native_validation_summary.csv`.
- [ ] Investigate rows with low confidence or candidate/native disagreement.

## Suggested pilot size

Start with 10 utterances. This keeps the first review manageable and lets us check whether the instructions make sense before scaling to the full 50-sample dev set.
