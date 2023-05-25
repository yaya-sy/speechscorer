"""Module implementing the different type of\
    masked language model based scorers"""

from ..base_scorer import BaseScorer
from typing import Union
from pathlib import Path

from torch.nn import functional as F
from torch import Tensor


class BaseMLMScorer(BaseScorer):
    """
    Will compute different scores from speech models trained\
    on masked language model objective.
    This is an abstract class for models like HuBERT\
    and derivatives (WavLM, etc.)

    This is highly inspired from BERT-like scoring as describe in\
    https://arxiv.org/pdf/1910.14659.pdf. The core idea is to mask\
    inputs one at team and to look how the model is hesitant at predicting\
    on those masked regions.
    """

    def __init__(self,
                 model_checkpoint: Union[Path, str],
                 use_cuda: bool=False):
        super().__init__(model_checkpoint=model_checkpoint, use_cuda=use_cuda)
        
    def cosine_similarity_scores(self,
                                 projected_hidden_vectors: Tensor,
                                 label_embeddings: Tensor,
                                 temperature: int) -> Tensor:
        """
        Computes the scores for each token (hidden vector) to belong to a label.
        The scores are computed using the cosine similarity.
        See the equation (3): https://arxiv.org/pdf/2106.07447.pdf.
        Both HuBERT and WavLM compute the scores the same way.

        Parameters
        ----------
        - projected_hidden_vectors: Tensor
            The hidden vectors already projected for computing the logits.
        - label_embeddings: Tensor
            The matrix containing an embeddings for each label.
        
        Returns
        -------
        - Tensor
            The consine similarity scores between hidden vectors and label embeddings.
        """

        projected_hidden_vectors_norm = F.normalize(projected_hidden_vectors, dim=-1)
        label_embeddings_norm = F.normalize(label_embeddings, dim=-1)
        scores = projected_hidden_vectors_norm @ label_embeddings_norm.T
        scores /= temperature
        return scores
