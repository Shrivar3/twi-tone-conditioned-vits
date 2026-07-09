# Full VITS 200-step sanity evaluation summary

Status:
- Full-VITS tone-conditioned and no-tone 200-step sanity checkpoints were evaluated with the ASR/WER pipeline.
- The same fixed 50-sample development set was used for paired comparison.
- ASR transcripts, per-sample WER/CER files, WER summaries, and Wilcoxon comparison results were generated.

Main result:
- Baseline mean WER: 1.008
- Full-VITS tone 200 mean WER: 1.968
- Full-VITS no-tone 200 mean WER: 2.288

Interpretation:
- The 200-step full-VITS models do not outperform the baseline.
- The tone-conditioned 200-step model is better than the no-tone 200-step model, but both are still much worse than baseline.
- This should be treated as an infrastructure/evaluation sanity result, not a final model-quality result.

Important caveat:
A 200-step VITS run is far too short to produce a production-quality TTS model. The value of this run is that it verifies the full training, inference, ASR/WER, and Wilcoxon pipeline end-to-end.

Generated locally but not committed:
- checkpoints/
- outputs/
- validation sample WAVs
- generated synthesis WAVs
- zip files
