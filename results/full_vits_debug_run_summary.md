# Full VITS debug run summary

Status:
- Full tone-conditioned VITS training path was pulled and executed on GPU.
- Filelists were prepared from the 50-sample development manifest.
- The debug full-tone run completed using configs/full_vits_tone_debug.yaml.
- The no-tone ablation debug run also completed using --disable-tone-conditioning.

Important caveat:
This was only a short debug run to verify the full VITS objective, dataloader, generator, discriminator, checkpointing, and validation-sample writing. It is not a trained production-quality model.

Generated locally but not committed:
- checkpoints/full_vits_tone_debug/
- checkpoints/full_vits_no_tone_debug/
- validation sample WAVs

Next technical step:
Run a longer controlled full-VITS experiment only after confirming validation samples and losses look sane.
