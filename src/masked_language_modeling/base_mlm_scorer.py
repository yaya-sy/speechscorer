"""Module implementing the different type of scorers"""

from base_scorer import BaseScorer
from typing import Union
from pathlib import Path

from torch.nn import functional as F
from torch import Tensor


class BaseMLMScorer(BaseScorer):
    """
    Will compute different scores from speech models trained\
    on masked language model objective.

    This is highly inspired from BERT-like scoring as describe in\
    https://arxiv.org/pdf/1910.14659.pdf. The core idea is to mask\
    inputs one at team and to look how the model is hesitant at predicting\
    on those masked regions.

    Two approaches are proposed. The first one require targets labels and\
    compute the (pseudo-)likelihood of the input speech (i.e the probability\
    of generating the right labels under the estimated parameters of the model.)\
    The second approache does'nt require any labels and uses only the probability\
    distribution over the labels to retrieve the metrics (entropy, perplexity).

    """
    def __init__(self,
                 path_to_checkpoint: Union[Path, str]):
        super().__init__(path_to_checkpoint)
        
    def cosine_similarity_scores(self,
                                 projected_hidden_vectors: Tensor,
                                 label_embeddings: Tensor,
                                 temperature: int
                                 ) -> Tensor:
        """
        Computes the scores for each token (hidden vector) to belong to a label.
        The scores are computed using the cosine similarity.
        See the equation (3): https://arxiv.org/pdf/2106.07447.pdf
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
    
    def cross_entropy(self,
                      scores: Tensor,
                      gold_labels: Tensor,
                      reduction: str="mean") -> Tensor:
        """
        Computes the cross entropy
        Parameters
        ----------
        - scores: Tensor
            The scores of each token to belong to each class.
        - gold_labels: Tensor
            The gold lables of each token in the sequence. Shape=[vocab_size]
        
        Returns
        -------
        - float:
            The cross-entropy of the labels, i.e the negative logprobability of the gold labels.
        """
        return F.cross_entropy(scores, gold_labels, reduction=reduction)