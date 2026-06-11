# Farmerline Week 1 Native Validation Checklist

## Goal

Validate the pilot tone annotation sheet with native Twi/Akan speakers before treating any tone labels as reference data.

## Files to share with reviewers

```text
data/manifests/native_validation_token_sheet.csv
docs/native_validation_protocol.md
```

## Reviewer requirements

Ideal reviewers should:

- be native or highly fluent Twi/Akan speakers;
- be comfortable reading the orthography used in the dataset;
- be willing to mark uncertainty rather than guess;
- provide short comments for ambiguous cases.

## Minimum pilot target

For the first pilot, aim for:

- 2-3 reviewers;
- 10 utterances;
- token-level tone labels;
- comments on ambiguity or orthographic issues.

## What to check after review

After reviewer sheets are returned, check:

- number of reviewed tokens;
- number of skipped tokens;
- number of `UNK` labels;
- agreement with candidate labels;
- disagreement between reviewers, if multiple reviewers are used;
- recurring causes of uncertainty.

## Follow-up decision

After the pilot, decide whether to:

1. continue Gemini-assisted annotation with native validation;
2. switch to a lexicon/rule-based workflow;
3. ask Farmerline for a higher-quota API key or existing tone resources;
4. reduce the labelled subset to a smaller high-quality development set.
