# Copyright (c) Fudan University.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms
# of the GNU General Public License version 3.

# llama tokenizer
import os

from typing import List, Union, Optional

import torch
from transformers import AutoTokenizer

try:
    from sentencepiece import SentencePieceProcessor
except:
    SentencePieceProcessor = None

class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        if SentencePieceProcessor is None:
            raise ModuleNotFoundError(
                "Detected sentencepiece not install. "
                "See https://github.com/google/sentencepiece"
            )
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        if self.bos_id in t:
            t = t[t.index(self.bos_id):]
        if self.eos_id in t:
            t = t[:t.index(self.eos_id)+1]
        return self.sp_model.decode(t)

    def batch_decode(self, batch_t: List[int]) -> str:
        for t in batch_t:
            if self.bos_id in t:
                t = t[t.index(self.bos_id):]
            if self.eos_id in t:
                t = t[:t.index(self.eos_id) + 1]
        return self.sp_model.decode(batch_t)


class MyTokenizer:
    """Masked tokenizer of hugging face to be similar to the one of meta,
    just used for testing purposes.
    """

    def __init__(self, model_path: Optional[str] = None):

        if model_path is None:
            self.sp_model = AutoTokenizer.from_pretrained("gpt2")
        else:
            self.sp_model = AutoTokenizer.from_pretrained(model_path)

        self.n_words = self.sp_model.vocab_size
        self.bos_id = self.sp_model.bos_token_id
        self.eos_id = self.sp_model.eos_token_id
        self.pad_id = self.sp_model.eos_token_id

    def encode(
        self,
        s: str,
        bos: bool = True,
        eos: bool = True,
        truncation: bool = True,
    ) -> List[int]:
        output = self.sp_model.encode(s, truncation=truncation)
        t = list(output)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        #input = torch.as_tensor(t)
        input = t.tolist()
        print(input)
        output = self.sp_model.decode(input)
        return output


class HFLikeTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

        # assign attributes from real tokenizer to masked one
        self.pad_id = self.tokenizer.pad_id
        self.eos_id = self.tokenizer.eos_id
        self.bos_id = self.tokenizer.bos_id

        # mask attribute to be similar to hugging face
        self.eos_token_id = self.tokenizer.eos_id
        self.pad_token_id = self.tokenizer.pad_id

        # to match hugging face attribute
        self.pad_token_id = self.pad_id

    def create_sequence_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = torch.where(
            tokens == self.tokenizer.pad_id,
            torch.zeros_like(tokens),
            torch.ones_like(tokens),
        )
        mask = torch.where(
            tokens == self.tokenizer.bos_id, torch.zeros_like(tokens), mask
        )
        mask = torch.where(
            tokens == self.tokenizer.eos_id, torch.zeros_like(tokens), mask
        )
        return mask

    def __call__(self, texts: Union[List[str], str], *args, **kwargs):
        if isinstance(texts, str):
            text = self.tokenizer.encode(texts, bos=True, eos=True)
            tokens = torch.tensor(text).long()
            mask = torch.ones_like(tokens)
        else:
            texts = [
                self.tokenizer.encode(text, bos=True, eos=True)
                for text in texts
            ]
            max_len = max(len(text) for text in texts)
            tokens = torch.full(
                (len(texts), max_len), self.tokenizer.pad_id
            ).long()
            for i, text in enumerate(texts):
                tokens[i, -len(text) :] = torch.tensor(  # noqa E203
                    text
                ).long()

            # TODO: decide how eos and bos should be handled - i need to mask
            # them? or not?
            mask = self.create_sequence_mask(tokens)
            for i in range(tokens.shape[0]):
                current_tokens = tokens[i, mask[i] == 1]
                tokens[
                    i, -len(current_tokens) - 1 : -1  # noqa E203
                ] = current_tokens
            mask = self.create_sequence_mask(tokens)

            # convert `pad_id` from -1 to 0, otherwise embedding will cause out
            # of bounds.
            tokens = torch.where(
                tokens == self.tokenizer.pad_id,
                torch.zeros_like(tokens),
                tokens,
            )
        output = {
            "input_ids": tokens,
            "attention_mask": mask,
        }
        return output

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            # llama's tokenizer only accepts List[int]
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def batch_decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.batch_decode(tokens)

