"""Module implementing abstract conditional language model based scorer."""
from ..base_scorer import BaseScorer
from logging import Logger
from typing import Union
from pathlib import Path
import torch

class BaseConditionalLanguageModelScorer(BaseScorer):
    """
    Base scorer for conditional language model speech models.\
    The logits are the scores over the vocabulary\
    at each step of the prediction. So computing the entropy on those\
    scores can be seen as the hesitation of the model on its predictions.
    """
    def __init__(self, model_checkpoint: Union[Path, str], use_gpu: bool = False):
        super().__init__(model_checkpoint, use_gpu)

    def load_model(self,
                   model_class,
                   logger: Logger) -> None:
        logger.info("Loading the model...")
        self.model = model_class.from_pretrained(self.model_checkpoint).eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu')
        logger.info(f"Using device {self.device}")
        self.model.to(self.device)