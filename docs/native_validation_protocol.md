# Native-Speaker Tone Validation Protocol

## Purpose

This protocol validates candidate Twi/Akan tone annotations for the Week 1 tone-annotated development set.

Candidate tone labels from Gemini or heuristic rules are not ground truth. They are suggestions to speed up review. Native-speaker labels should be treated as the reference labels for evaluation and later model conditioning.

## English gloss and reviewer notes update

The native validation sheet now includes automatic English glosses to help reviewers understand the intended meaning of each Twi/Akan utterance.

These glosses are only reviewer aids. They are not final translations and should not overwrite the original Twi/Akan text.

The relevant columns are:

| Column | Purpose |
|---|---|
| `content_text_twi` | Original Twi/Akan text from the dataset. Do not edit this column. |
| `content_text_english_gloss_auto` | Automatic English gloss of the full Twi/Akan sentence. Use this to understand the intended meaning. |
| `token` | The Twi/Akan token being reviewed. |
| `token_english_gloss_auto` | Automatic English gloss for the individual token, where available. |
| `candidate_tone` | Gemini's candidate tone label. |
| `candidate_reason` | Gemini's short reason for the candidate tone label. |
| `reviewer_note_prompt` | A helper prompt explaining what the reviewer should check. |
| `native_tone_label` | Native speaker's corrected tone label. |
| `native_corrected_token` | Corrected native token if the token is wrong, unnatural, or badly split. |
| `native_corrected_twi_sentence` | More natural native Twi/Akan wording if the full sentence should be corrected. |
| `native_english_meaning_note` | Native speaker's correction or clarification of the English meaning. |
| `reviewer_notes` | Main free-text reviewer note column. Use this especially for meaning, wording, and native-language corrections. |

Reviewers should use `reviewer_notes` especially when:

- the automatic English gloss is wrong or incomplete;
- the Twi/Akan wording is unnatural;
- there is a better native word or phrase;
- the tokenisation is wrong;
- the tone depends on context;
- the candidate tone label is wrong;
- the item needs discussion before it can be used as a reliable training label.

The most important fields for reviewer feedback are:

1. `native_tone_label`
2. `native_english_meaning_note`
3. `native_corrected_twi_sentence`
4. `reviewer_notes`

The `reviewer_notes` column should be treated as the main place for qualitative feedback from the native speaker.

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
