# Native-Speaker Tone Validation Protocol

## Purpose

This protocol validates candidate Twi/Akan tone annotations for the Week 1 tone-annotated development set.

Candidate tone labels from Gemini or heuristic rules are not ground truth. They are suggestions to speed up review. Native-speaker labels should be treated as the reference labels for evaluation and later model conditioning.

## Input files

```text
data/manifests/dev_set.csv
data/manifests/tone_annotated_dev.csv
data/manifests/gemini_tone_annotated_dev.csv
```

The Gemini file is optional. If unavailable, the reviewer sheet can still be created from the dev set and conservative annotation file.

## Generated reviewer file

```text
data/manifests/native_validation_token_sheet.csv
```

## Summary output after review

```text
results/native_validation_summary.csv
```

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

If a reviewer is uncertain, `UNK` is acceptable. It is better to mark uncertainty honestly than to create overconfident tone labels.

When multiple reviewers disagree, the disagreement should be recorded rather than hidden. Those rows can be escalated for adjudication by a stronger native-speaker reviewer or linguistic expert.
