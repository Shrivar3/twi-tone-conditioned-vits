# Full VITS 200-step sanity evaluation summary

Status:
- Full-VITS tone-conditioned and no-tone 200-step sanity checkpoints were evaluated using the existing ASR/WER pipeline.
- The same fixed development set was used for paired comparison.
- ASR transcripts, per-sample WER/CER files, WER summaries, and Wilcoxon comparison results were generated.

Important caveat:
This is a 200-step sanity run, not a fully trained production-quality model. The purpose is to verify that full-VITS training, inference, checkpoint loading, synthesis, ASR/WER, and paired comparison can run end-to-end.

Interpretation:
- These results should be treated as infrastructure/evaluation sanity results.
- A longer controlled full-VITS training run would be required before making claims about final TTS quality.
- MOS remains pending unless human listening ratings are collected.

Generated locally but not committed:
- checkpoints/
- outputs/
- validation sample WAVs
- generated synthesis WAVs
