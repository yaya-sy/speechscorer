"""Module implementing abstract conditional language model based scorer."""
from ..base_scorer import BaseScorer
from typing import Union
from pathlib import Path

class BaseCLMScorer(BaseScorer):
    """
    Base scorer for conditional language model speech models.\
    The logits are the scores over the vocabulary\
    at each step of the prediction. So computing the entropy on those\
    scores can be seen as the hesitation of the model on its predictions.
    """
    def __init__(self, model_checkpoint: Union[Path, str], use_cuda: bool = False):
        super().__init__(model_checkpoint, use_cuda)