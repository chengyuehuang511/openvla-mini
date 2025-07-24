from transformers import LogitsProcessor
import torch

class ActionTokenFilter(LogitsProcessor):
    def __init__(self, valid_ids: torch.LongTensor):
        self.valid_ids = valid_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        mask = torch.full_like(scores, -1e8)
        mask[:, self.valid_ids] = 0
        return scores + mask