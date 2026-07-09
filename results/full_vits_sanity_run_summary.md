# Full VITS sanity run summary

Status:
- Full tone-conditioned VITS training path was implemented and pulled onto the GPU instance.
- Full-VITS filelists were prepared from the development manifest.
- The short debug run completed successfully.
- A longer 200-step sanity run completed successfully.
- A no-tone ablation sanity run was also run if present locally.

What this verifies:
- Full VITS-style dataloader works.
- Tone-aware text prior path runs.
- Posterior encoder / flow / duration / mel / adversarial / feature-matching training loop executes.
- Generator and discriminator optimisation steps run on GPU.
- Checkpoint writing works.
- Validation sample writing works.

Important caveat:
This is still only a sanity run, not a trained production-quality TTS model. The 200-step run is intended to verify the full objective and training infrastructure, not to claim final audio quality.

Generated locally but not committed:
- checkpoints/full_vits_tone_debug/
- checkpoints/full_vits_no_tone_debug/
- checkpoints/full_vits_tone_debug_200/
- checkpoints/full_vits_no_tone_debug_200/
- validation sample WAVs

Next technical step:
Inspect validation samples and loss behaviour. If they look sane, run a longer controlled full-VITS experiment and then evaluate with the existing WER/MOS/Wilcoxon pipeline.
