"""Whisper conditional language model based scorer."""
from .base_clm_scorer import BaseConditionalLanguageModelScorer
from typing import Union, Optional, Dict
from pathlib import Path
import logging

from transformers import  WhisperForConditionalGeneration
from torch import Tensor
import torch

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class WhisperConditionalLanguageModelScorer(BaseConditionalLanguageModelScorer):
    """
    This class implements a Whisper conditional language model\
    based scorer.
    """
    def __init__(self, model_checkpoint: Union[Path, str], use_gpu: bool = False):
        super().__init__(model_checkpoint, use_gpu)
        self.load_model(model_class=WhisperForConditionalGeneration, logger=LOGGER)
    
    def forward_model(self,
                      x: Tensor,
                      **kwargs
                      ) -> Dict[str, Tensor]:
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model.generate(inputs=x,
                                         return_dict_in_generate=True,
                                         output_scores=True,
                                         num_beams=1,
                                         do_sample=False,
                                         max_length=80).scores
            logits = torch.stack(logits, 1)
            logits[logits == float("-Inf")] = 1e-12

        return logits