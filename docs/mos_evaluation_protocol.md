# MOS Evaluation Protocol

## Purpose

This evaluation measures the perceived naturalness and intelligibility of the baseline Farmerline Twi TTS checkpoint.

## Samples

Use the fixed Week 1 development set:

```text
data/manifests/dev_set.csv
```

Generated baseline audio should be stored under:

```text
outputs/baseline_tts/wavs/
```

Do not commit generated audio to GitHub.

## Raters

Suggested pilot:

- 3-5 native Twi/Akan speakers.

Suggested full evaluation:

- 20 native Twi/Akan speakers, if available.

## Rating questions

For each audio sample, ask two questions.

### Naturalness

How natural does the speech sound?

1 = very unnatural  
2 = somewhat unnatural  
3 = acceptable but clearly synthetic  
4 = mostly natural  
5 = completely natural

### Intelligibility

How easy is it to understand?

1 = impossible to understand  
2 = difficult to understand  
3 = partly understandable  
4 = mostly understandable  
5 = perfectly clear

## Raw results format

Store raw ratings in:

```text
results/baseline_mos_raw.csv
```

Required columns:

```text
rater_id,utt_id,naturalness_score,intelligibility_score,comments
```

## Summary output

The summary script writes:

```text
results/baseline_mos_summary.csv
```

Metrics:

- number of ratings;
- number of raters;
- number of samples;
- mean naturalness MOS;
- standard deviation naturalness MOS;
- mean intelligibility MOS;
- standard deviation intelligibility MOS.
