"""Whisper conditional language model based scorer."""
from .base_clm_scorer import BaseCLMScorer
from typing import Union, Optional, Dict
from pathlib import Path
import logging

from transformers import  WhisperForConditionalGeneration
from torch import Tensor
import torch

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class WhisperCLMScorer(BaseCLMScorer):
    """
    This class implements a Whisper conditional language model\
    based scorer.
    """
    def __init__(self, model_checkpoint: Union[Path, str], use_cuda: bool = False):
        super().__init__(model_checkpoint, use_cuda)

    def load_model(self, model_checkpoint: Union[Path, str]) -> None:
        LOGGER.info("Loading the model...")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint).eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        LOGGER.info(f"Using device {self.device}")
        self.model.to(self.device)
    
    def forward_model(self,
                      x: Tensor,
                      labels: Optional[Tensor] = None
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