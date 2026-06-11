# Native-Speaker Tone Validation Protocol

## Purpose

This protocol validates candidate Twi/Akan tone annotations for the Week 1 tone-annotated development set. The validation set should be reviewed by native Twi/Akan speakers before the labels are treated as reference data.

## Files

Input files:

- `data/manifests/dev_set.csv`
- `data/manifests/gemini_tone_annotated_dev.csv` if available
- `data/manifests/tone_annotated_dev.csv` if available

Generated reviewer file:

- `data/manifests/native_validation_token_sheet.csv`

Summary output after review:

- `results/native_validation_summary.csv`

## Tone labels

Use the following labels:

- `H`: high tone
- `L`: low tone
- `F`: falling tone
- `H-L`, `L-H`, etc.: multi-syllable or contour sequence where needed
- `UNK`: uncertain or not reliably inferable

## Reviewer instructions

For each row in `native_validation_token_sheet.csv`:

1. Read the full sentence in `context_text`.
2. Look at the target `token`.
3. Check the candidate tone label in `candidate_tone`.
4. Fill in `native_tone_label` with the correct tone label.
5. Fill in `native_confidence` from 1 to 5:
   - 1 = very unsure
   - 2 = unsure
   - 3 = moderately confident
   - 4 = confident
   - 5 = very confident
6. Set `review_status` to one of:
   - `reviewed`
   - `uncertain`
   - `skip`
7. Add comments in `reviewer_notes` if there is ambiguity, dialect variation, spelling uncertainty, or if the token needs a wider sentence context.

## Important notes

Candidate labels from Gemini or heuristic rules are not ground truth. They are only suggestions to speed up review. Native-speaker labels are the reference labels for evaluation and later model conditioning.

If a reviewer is uncertain, `UNK` is acceptable. It is better to mark uncertainty honestly than to create overconfident tone labels.
