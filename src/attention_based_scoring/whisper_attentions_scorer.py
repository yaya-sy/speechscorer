"""Whisper conditional language model based scorer."""
from .base_attention_scorer import BaseAttentionScorer
from typing import Union, Optional, Dict
from pathlib import Path
import logging

from transformers import  WhisperForConditionalGeneration
from torch import Tensor
import torch

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class WhisperAttentionScorer(BaseAttentionScorer):
    """This class implements a Whisper attention based scorer."""
    def __init__(self, model_checkpoint: Union[Path, str], use_cuda: bool = False):
        super().__init__(model_checkpoint, use_cuda)

    def load_model(self, model_checkpoint: Union[Path, str]) -> None:
        LOGGER.info("Loading the model...")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        LOGGER.info(f"Using device {self.device}")
        self.model.to(self.device)
    
    def forward_model(self,
                      x: Tensor,
                      labels: Optional[Tensor] = None
                      ) -> Dict[str, Tensor]:
        with torch.no_grad():
            x = x.to(self.device)
            generation_outputs = self.model.generate(inputs=x,
                                                     return_dict_in_generate=True,
                                                     output_scores=True,
                                                     output_attentions=True,
                                                     num_beams=1,
                                                     do_sample=False,
                                                     max_length=50)
            # encoder_attentions = torch.stack(generation_outputs.encoder_attentions, 1)
            cross_attentions = []
            for token in generation_outputs.cross_attentions:
                cross_attentions.append(torch.stack(token, 1))
            # [batch, layers, heads, tgt, src]
            cross_attentions = torch.cat(cross_attentions, -2)

        return {"encoder_attentions": None,
                "cross_attentions": cross_attentions,
                "logits": None,
                "loss": None,
                "labels": labels}