# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vendored from ppocr/postprocess/rec_postprocess.py — CTC decode (numpy only)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


class BaseRecLabelDecode:
    """Convert between text labels and text indices."""

    def __init__(
        self,
        character_dict_path: str | Path | None = None,
        use_space_char: bool = False,
    ) -> None:
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str: list[str] = []

        if character_dict_path is None:
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")
            dict_character = list(self.character_str)
        else:
            dict_path = Path(character_dict_path)
            with dict_path.open("rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    self.character_str.append(line.decode("utf-8").strip("\n").strip("\r\n"))
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in str(dict_path):
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character

    def pred_reverse(self, pred: str) -> str:
        pred_re: list[str] = []
        c_current = ""
        for c in pred:
            if not bool(re.search(r"[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)
        return "".join(pred_re[::-1])

    def add_special_char(self, dict_character: list[str]) -> list[str]:
        return dict_character

    def decode(
        self,
        text_index: np.ndarray,
        text_prob: np.ndarray | None = None,
        is_remove_duplicate: bool = False,
        return_word_box: bool = False,
    ) -> list[tuple[str, float] | tuple[str, float, list[Any]]]:
        result_list: list[tuple[str, float] | tuple[str, float, list[Any]]] = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [self.character[text_id] for text_id in text_index[batch_idx][selection]]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = np.ones(np.count_nonzero(selection))
            if len(conf_list) == 0:
                conf_list = np.array([0.0])

            text = "".join(char_list)

            if self.reverse:
                text = self.pred_reverse(text)

            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(text, selection)
                result_list.append(
                    (
                        text,
                        float(np.mean(conf_list)),
                        [
                            len(text_index[batch_idx]),
                            word_list,
                            word_col_list,
                            state_list,
                        ],
                    )
                )
            else:
                result_list.append((text, float(np.mean(conf_list))))
        return result_list

    def get_word_info(
        self, text: str, selection: np.ndarray
    ) -> tuple[list[list[str]], list[list[int]], list[str]]:
        state: str | None = None
        word_content: list[str] = []
        word_col_content: list[int] = []
        word_list: list[list[str]] = []
        word_col_list: list[list[int]] = []
        state_list: list[str] = []
        valid_col = np.where(selection)[0]

        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            elif bool(re.search(r"[\w]", char, re.UNICODE)) and char != "_":
                c_state = "en&num"
            else:
                c_state = "splitter"

            if char == "'" and state == "en&num":
                c_state = "en&num"

            if (
                char == "."
                and state == "en&num"
                and c_i + 1 < len(text)
                and bool(re.search(r"[0-9]", text[c_i + 1]))
            ):
                c_state = "en&num"
            if char == "-" and state == "en&num":
                c_state = "en&num"

            if state is None:
                state = c_state

            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            if state != "splitter":
                word_content.append(char)
                word_col_content.append(int(valid_col[c_i]))

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return word_list, word_col_list, state_list

    def get_ignored_tokens(self) -> list[int]:
        return [0]


class CTCLabelDecode(BaseRecLabelDecode):
    """Decode CTC predictions into text labels."""

    def __init__(
        self,
        character_dict_path: str | Path | None = None,
        use_space_char: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(character_dict_path, use_space_char)

    def __call__(
        self,
        preds: np.ndarray | tuple[np.ndarray, ...] | list[np.ndarray],
        label: np.ndarray | None = None,
        return_word_box: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[str, float]] | tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        del args, kwargs
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True,
            return_word_box=return_word_box,
        )
        if label is None:
            return text
        label_text = self.decode(label)
        return text, label_text

    def add_special_char(self, dict_character: list[str]) -> list[str]:
        return ["blank", *dict_character]
