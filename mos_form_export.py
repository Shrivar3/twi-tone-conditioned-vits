from __future__ import annotations

import re
from pathlib import Path

import yaml

DEFAULT_PRESERVE = r"a-zA-Z0-9ɛƐɔƆáàâāéèêēíìîīóòôōúùûūńǹḿm̀ḿǹń'’ "


def load_normalisation_config(path: str | Path | None = None) -> dict:
    if path is None:
        return {
            "normalisation": {
                "lowercase": True,
                "strip_punctuation": True,
                "collapse_whitespace": True,
                "preserve_characters": DEFAULT_PRESERVE,
            }
        }
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalise_text(text: str, config: dict | None = None) -> str:
    """Normalise Twi text for WER/CER comparison.

    This keeps common Akan/Twi characters such as ɛ and ɔ while removing most
    punctuation. Keep this conservative and document all changes in reports.
    """
    cfg = (config or load_normalisation_config())["normalisation"]
    s = str(text).strip()

    if cfg.get("lowercase", True):
        s = s.lower()

    if cfg.get("strip_punctuation", True):
        preserve = cfg.get("preserve_characters", DEFAULT_PRESERVE)
        s = re.sub(fr"[^{preserve}]", " ", s)

    if cfg.get("collapse_whitespace", True):
        s = re.sub(r"\s+", " ", s).strip()

    return s
