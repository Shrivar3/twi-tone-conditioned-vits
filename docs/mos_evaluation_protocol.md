# MOS Evaluation Protocol

## Purpose

This evaluation measures the baseline naturalness and intelligibility of the current Farmerline Twi TTS checkpoint.

## Samples

Use the fixed Week 1 dev set:

- `data/manifests/dev_set.csv`
- 50 utterances
- Audio generated from the baseline checkpoint should be placed under `outputs/baseline_tts/wavs/`.

## Raters

Target:

- Pilot: 3-5 native Twi speakers
- Full evaluation: 20 native Twi speakers

## Rating Questions

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

## Raw Results Format

Store raw ratings in `results/baseline_mos_raw.csv`.

Required columns:

- `rater_id`
- `utt_id`
- `naturalness_score`
- `intelligibility_score`
- `comments`

## Summary Output

The summary script writes `results/baseline_mos_summary.csv`.

Metrics:

- number of ratings
- number of raters
- number of samples
- mean naturalness MOS
- standard deviation naturalness MOS
- mean intelligibility MOS
- standard deviation intelligibility MOS
