from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.full_vits.config import AudioConfig, ModelConfig
from src.training.full_vits.data import FullVitsBatchCollator, FullVitsFilelistDataset
from src.training.full_vits.models import MultiPeriodDiscriminator, ToneConditionedVitsGenerator
from src.training.full_vits.train_loop import training_step


class TinyTokenizer:
    def __init__(self) -> None:
        chars = ["<pad>", " ", "a", "e", "h", "k", "m", "o", "s", "t", "w", "y"]
        self.vocab = {char: idx for idx, char in enumerate(chars)}
        self.pad_token_id = 0

    def __len__(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def __call__(self, text: str, return_attention_mask: bool = True):
        ids = [self.vocab.get(ch, 1) for ch in str(text).lower() if ch in self.vocab]
        if not ids:
            ids = [self.pad_token_id]
        out = {"input_ids": ids}
        if return_attention_mask:
            out["attention_mask"] = [1] * len(ids)
        return out


def tiny_audio_config() -> AudioConfig:
    return AudioConfig(
        sampling_rate=4000,
        filter_length=128,
        hop_length=16,
        win_length=128,
        n_mel_channels=16,
    )


def tiny_model_config(tokenizer: TinyTokenizer) -> ModelConfig:
    return ModelConfig(
        n_vocab=len(tokenizer),
        num_tones=11,
        spec_channels=65,
        segment_size=256,
        inter_channels=16,
        hidden_channels=16,
        filter_channels=32,
        n_heads=2,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3]],
        upsample_rates=[4, 4],
        upsample_initial_channel=32,
        upsample_kernel_sizes=[8, 8],
        tone_conditioning_mode="concat_projection",
        tone_embedding_scale=1.0,
        discriminator_base_channels=2,
        discriminator_periods=[2, 3],
    )


class FullVitsSmokeTest(unittest.TestCase):
    def make_batch(self):
        tokenizer = TinyTokenizer()
        audio_config = tiny_audio_config()

        with tempfile.TemporaryDirectory() as tmp:
            filelist = Path(tmp) / "train.txt"
            filelist.write_text(
                "missing_a.wav|me ka|H L\n"
                "missing_b.wav|wo ho ye|L H H\n",
                encoding="utf-8",
            )

            dataset = FullVitsFilelistDataset(
                filelist,
                tokenizer=tokenizer,
                audio_config=audio_config,
                allow_missing_audio=True,
                synthetic_audio_seconds=0.3,
            )
            loader = DataLoader(
                dataset,
                batch_size=2,
                collate_fn=FullVitsBatchCollator(text_pad_id=tokenizer.pad_token_id),
            )
            batch = next(iter(loader))

        return tokenizer, audio_config, batch

    def test_one_batch_dataloader(self):
        _, _, batch = self.make_batch()

        self.assertEqual(batch["text_ids"].shape[0], 2)
        self.assertEqual(batch["tone_ids"].shape, batch["text_ids"].shape)
        self.assertEqual(batch["spectrograms"].ndim, 3)
        self.assertEqual(batch["waveforms"].ndim, 2)

    def test_generator_and_discriminator_forward(self):
        tokenizer, _, batch = self.make_batch()
        model_config = tiny_model_config(tokenizer)
        generator = ToneConditionedVitsGenerator(model_config)
        discriminator = MultiPeriodDiscriminator.from_model_config(model_config)

        outputs = generator(
            text_ids=batch["text_ids"],
            text_lengths=batch["text_lengths"],
            tone_ids=batch["tone_ids"],
            spectrogram=batch["spectrograms"],
            spec_lengths=batch["spec_lengths"],
        )

        self.assertIn("y_hat", outputs)
        self.assertEqual(outputs["y_hat"].shape[0], 2)

        y = batch["waveforms"][:, None, : outputs["y_hat"].shape[-1]]
        y_hat = outputs["y_hat"][..., : y.shape[-1]]
        disc_out = discriminator(y, y_hat)
        self.assertEqual(len(disc_out), 4)

    def test_one_tiny_training_step(self):
        tokenizer, audio_config, batch = self.make_batch()
        model_config = tiny_model_config(tokenizer)
        generator = ToneConditionedVitsGenerator(model_config)
        discriminator = MultiPeriodDiscriminator.from_model_config(model_config)
        optimizer_g = torch.optim.AdamW(generator.parameters(), lr=1e-4)
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)

        metrics = training_step(
            batch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            audio_config=audio_config,
            loss_weights={
                "mel": 1.0,
                "kl": 1.0,
                "duration": 1.0,
                "feature_matching": 1.0,
                "generator_adv": 1.0,
            },
            grad_clip=1.0,
        )

        self.assertGreater(metrics["loss_g"], 0.0)
        self.assertGreater(metrics["loss_d"], 0.0)


if __name__ == "__main__":
    unittest.main()
