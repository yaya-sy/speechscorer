"""WavLM conditional language model based scorer."""
from .base_clm_scorer import BaseConditionalLanguageModelScorer
from typing import Union, Dict
from pathlib import Path
import logging

from transformers import  HubertForCTC
from torch import Tensor
import torch

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class HuBERTConditionalLanguageModelScorer(BaseConditionalLanguageModelScorer):
    """
    This class implements a WavLM conditional language model\
    based scorer.
    """
    def __init__(self, model_checkpoint: Union[Path, str], use_cuda: bool = False):
        super().__init__(model_checkpoint, use_cuda)
        self.load_model(model_class=HubertForCTC, logger=LOGGER)
    
    def forward_model(self,
                      x: Tensor
                      ) -> Dict[str, Tensor]:
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x).logits
        return logits