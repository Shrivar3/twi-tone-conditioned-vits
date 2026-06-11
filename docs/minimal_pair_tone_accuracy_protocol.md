# Minimal-Pair Tone Accuracy Protocol

## Purpose

This test measures whether the TTS system preserves lexical tone contrasts in Twi/Akan.

## Why this is separate from MOS

MOS measures perceived naturalness and intelligibility. Minimal-pair tone accuracy measures whether listeners hear the intended tone contrast.

## Test Construction

Create a file:

- `data/minimal_pairs/minimal_pairs.csv`

Use the template:

- `data/minimal_pairs/minimal_pairs_template.csv`

Each minimal pair should be checked by a native Twi/Akan speaker.

## Required Columns

- `pair_id`: shared ID for the minimal pair
- `item_id`: A or B
- `text`: full prompt sentence or phrase
- `target_word`: target word being tested
- `gloss`: English meaning
- `tone_pattern`: expected tone pattern, e.g. H, L, F, H-L
- `contrast_group`: optional category for the contrast
- `notes`: explanation or uncertainty
- `native_checked`: yes/no

## Evaluation

For each generated audio sample, native speakers answer a forced-choice question:

- Which word/meaning did you hear?
- A
- B
- Unsure

Tone accuracy is:

```text
correct forced-choice responses / total non-unsure responses
```

Report:

- total judgments
- number correct
- number unsure
- tone accuracy percentage
