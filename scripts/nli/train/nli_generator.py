import math
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.ngram_repeat_block import NGramRepeatBlock


class NLIGenerator(nn.Module):
    def __init__(
        self,
        model,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (~fairseq.models.FairseqModel): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.eos = 999


    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)


    @torch.no_grad()
    def generate(
        self, model, sample: Dict[str, Dict[str, Tensor]], **kwargs
    ) -> List[List[Dict[str, Tensor]]]:
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
        """
        with torch.no_grad():
            labels = self.model(**sample["net_input"]).max(-1)[0].cpu()
        finalized = []

        for b in range(labels.size(0)):
            finalized.append([{
                "tokens": torch.tensor([labels[b].item()]),
                "alignment": None,
                "steps": None,
                "positional_scores": None,
                "score": None,
                "hypo_attn": None,
                }])

        return finalized
