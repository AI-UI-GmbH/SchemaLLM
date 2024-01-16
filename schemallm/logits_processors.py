import logging
import json
import torch
from typing import List
from transformers import PreTrainedTokenizer, StoppingCriteria, LogitsProcessor


class StringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if len(input_ids[0]) <= self.prompt_length:
            return False

        last_token_id = input_ids[0][-1]
        last_token = self.tokenizer.decode(last_token_id, skip_special_tokens=True)

        result = '"' in last_token

        return result


class NumberStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_length: int,
        precision: int = 3,
    ):
        self.tokenizer = tokenizer
        self.precision = precision
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        decoded = self.tokenizer.decode(input_ids[0][self.prompt_length :], skip_special_tokens=True)

        if decoded.count(".") > 1:
            return True

        if decoded.count(".") == 1 and len(decoded.strip().split(".")[1]) > self.precision:
            return True

        if len(decoded) > 1 and any(c.isdigit() for c in decoded) and decoded[-1] in [" ", "\n"]:
            return True

        return False


class NumberLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        vocab_size = len(tokenizer)
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for _, token_id in tokenizer.get_vocab().items():
            token_str = tokenizer.decode(token_id).strip()

            if token_str == "" or (all(c.isdigit() or c == "." for c in token_str) and token_str.count(".") <= 1):
                self.allowed_mask[token_id] = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        mask = self.allowed_mask.expand_as(scores)
        scores[~mask] = -float("inf")

        return scores


class LiteralLogitsProcessor(LogitsProcessor):
    def __init__(self, literals: List[str], tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.literals_tokens = [self.tokenizer.encode(literal, add_special_tokens=False) for literal in literals]

    def compute_mask(self):
        allowed_token_ids = [literal_tokens[0] for literal_tokens in self.literals_tokens if len(literal_tokens) > 0]
        vocab_size = len(self.tokenizer)
        allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)

        for token_id in allowed_token_ids:
            allowed_mask[token_id] = True
        allowed_mask[self.tokenizer.eos_token_id] = True

        if len(allowed_token_ids) == 0:
            allowed_mask = torch.ones(vocab_size, dtype=torch.bool)

        return allowed_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        allowed_mask = self.compute_mask()
        mask = allowed_mask.expand_as(scores)
        scores[~mask] = -float("inf")
        token_id = torch.argmax(scores[0])
        self.literals_tokens = [
            literal_tokens[1:]
            for literal_tokens in self.literals_tokens
            if len(literal_tokens) > 0 and literal_tokens[0] == token_id
        ]
        return scores
