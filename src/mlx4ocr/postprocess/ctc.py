"""CTC recognition post-processing."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mlx4ocr.types import TextRecognition


def load_character_dict(
    dict_path: Path,
    *,
    use_space_char: bool = True,
) -> tuple[str, ...]:
    """Load a PP-OCR character dictionary from disk.

    Args:
        dict_path: Path to a newline-delimited character file.
        use_space_char: Whether to append a space character to the dictionary.

    Returns:
        Character list with the CTC blank token prepended.
    """
    if not dict_path.is_file():
        raise FileNotFoundError(f"missing character dictionary: {dict_path}")

    characters: list[str] = []
    with dict_path.open(encoding="utf-8") as handle:
        for line in handle:
            characters.append(line.strip("\n").strip("\r\n"))
    if use_space_char:
        characters.append(" ")
    return ("blank", *characters)


def _pred_reverse(pred: str) -> str:
    pred_parts: list[str] = []
    current = ""
    for char in pred:
        if not re.search(r"[a-zA-Z0-9 :*./%+-]", char):
            if current:
                pred_parts.append(current)
            pred_parts.append(char)
            current = ""
        else:
            current += char
    if current:
        pred_parts.append(current)
    return "".join(pred_parts[::-1])


def ctc_decode(
    preds: np.ndarray,
    characters: Sequence[str],
    *,
    reverse: bool = False,
) -> tuple[TextRecognition, ...]:
    """Decode CTC softmax outputs into text labels.

    Args:
        preds: Softmax tensor ``[B, T, num_classes]``.
        characters: Character table with blank at index 0.
        reverse: Whether to reverse mixed Arabic-like predictions.

    Returns:
        One ``TextRecognition`` per batch item.
    """
    if preds.ndim != 3:
        raise ValueError(f"expected softmax [B, T, C], got shape {preds.shape}")

    indices = preds.argmax(axis=2)
    probs = preds.max(axis=2)
    results: list[TextRecognition] = []
    for batch_idx in range(indices.shape[0]):
        sequence = indices[batch_idx]
        confidence = probs[batch_idx]
        selection = np.ones(sequence.shape[0], dtype=bool)
        selection[1:] = sequence[1:] != sequence[:-1]
        selection &= sequence != 0

        char_list = [characters[int(text_id)] for text_id in sequence[selection]]
        conf_list = confidence[selection]
        if conf_list.size == 0:
            score = 0.0
        else:
            score = float(np.mean(conf_list))

        text = "".join(char_list)
        if reverse:
            text = _pred_reverse(text)
        results.append(TextRecognition(text=text, score=score))
    return tuple(results)
