"""Module implementing the different type of scorers"""

from typing import Union, Dict, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import logging

from torch.nn import functional as F
from torch import Tensor
import torch
import fairseq

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class Scorer(ABC):
    """

    """
    def __init__(self,
                 path_to_checkpoint: Union[Path, str]):

        self.model = self.load_model(path_to_checkpoint)
        self.cfg = None
    
    @abstractmethod
    def load_model(self, path_to_checkpoint: Union[Path, str]) -> None:
        pass

    @abstractmethod
    def forward_model(self, input: Tensor, batch_size: int=32):
        pass
        
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
    
    def label_probabilities(self, scores: Tensor) -> Tensor:
        """
        Normalizes the label scores in order to get the\
        probability distribution over the labels.

        Parameters
        ----------
        - scores: Tensor
            Tensor representing the score for each to belong to a label.
        
        Returns
        -------
        - Tensor:
            The probaility distribution over the labels.
        """
        return F.softmax(scores, dim=-1)
    
    def entropy(self, probabilities: Tensor) -> float:
        """
        Computes the casual entropy of a probability distribution.
        This metric tells us how much the model is hesidant in her predictions.
        However, the entropy will be low when the the model is confident in her\
        even if he's wrong. Cross entropy/log-likelihood is an alternative to this.

        Parameters
        ----------
        - probabilities: Tensor
            The probability distribution over labels.
        
        Returns
        -------
        - float:
            The entropy of the probability distribution, the uncertainty of the model.
        """
        # get a vector
        probabilities = probabilities.flatten()
        return -torch.sum(probabilities * torch.log(probabilities)).item()
    
    def cross_entropy(self,
                      scores: Tensor,
                      gold_labels: Tensor) -> float:
        """
        Computes the cross entropy

        Parameters
        ----------
        - scores: Tensor
            The scores of each token to belong to a class.
        - gold_labels: Tensor
            The gold lables of each token in the sequence. Shape=[vocab_size]
        
        Returns
        -------
        - float:
            The cross-entropy of the labels, i.e the negative logprobability of the gold labels.
        """
        return F.cross_entropy(scores, gold_labels).item()

    def loglikelihood(self, cross_entropy: float) -> float:
        """
        Computes the (log-)likelihhod of the labels given the cross-entropy.
        The loglihood is computed by just taking the opposit of the cross-entropy.
        
        Parameters
        ----------
        - cross_entropy: float
            The loglikelihood of the audio labels.
        """
        return -cross_entropy
    
    def perplexity(self, entropy_value: float) -> float:
        """Computes the perplexity from the entropy."""
        return torch.exp(entropy_value)

class HuBERTScorer(Scorer):

    def __init__(self, path_to_checkpoint):
        super().__init__(path_to_checkpoint=path_to_checkpoint)
    
    def load_model(self, path_to_checkpoint: Union[Path, str]) -> None:
        LOGGER.info(f"Loading model from {path_to_checkpoint}...")
        models, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([path_to_checkpoint])
        self.model = models[0].eval()
        if torch.cuda.is_available():
            LOGGER.info(f"CUDA is available! Let use it.")
            self.model.cuda()
        else:
            LOGGER.info(f"CUDA not founded. Model loaded on CPU.")
            self.model.cpu()
        self.cfg = cfg

    def forward_model(self, input: Tensor, batch_size: int = 32):
        with torch.no_grad() :
            # forward the convolution module in order to extract the features on the wavform
            features = self.model.feature_extractor(input)
            # apply HuBERT forward of these features
            features = features.transpose(1, 2)
            # features = hubert.layer_norm(features) # : Maybe not deactivate this even in eval mode
            features = self.model.post_extract_proj(features)
            ## Repeating the sequence so we can apply the masking on each token of the sequence.
            features = features.repeat(features.shape[1], 1, 1) # shape[seq_len, seq_len, embedding_dim]
            # here, we create the boolean masking matrix with the value 'True' in the diagonal.
            masker = torch.zeros((features.shape[0], features.shape[1]))
            masker.fill_diagonal_(1)
            masker = masker.bool()
            # apply mask to each token in the sequence.
            features[masker] = self.model.mask_emb
            hidden_vectors = []
            # run the BERT encoder
            for input_batch in features.split(batch_size):
                h, _ = self.model.encoder(input_batch)
                hidden_vectors.append(h)
            hidden_vectors = torch.cat(hidden_vectors)
            # get only the masked tokens
            masked_hidden_vectors = hidden_vectors[masker]
            # project the hidden vectors in order to be able to compute
            return self.model.final_proj(masked_hidden_vectors)

    def scores(self,
               input_wavform: Tensor,
               gold_labels: Optional[Tensor]=None,
               batch_size=32) -> Dict[str, str]:
        """
        Retrieves all the metrics for a given input_wavform

        Parameters
        ----------
        - input_wavform:
            Tensor representing the audio we want to score.
        
        Returns
        -------
        - dict:
            Pairing of metric names and their values.
        """
        projected_hidden_vectors = self.forward_model(input=input_wavform, batch_size=batch_size)
        cosine_similarity_scores = self.cosine_similarity_scores(projected_hidden_vectors, self.cfg.model.logit_temp)
        # computing all statstics
        probabilities = self.label_probabilities(cosine_similarity_scores)
        entropy = self.entropy(probabilities)
        perplexity = self.perplexity(entropy)
        
        metrics = {
            "entropy": entropy,
            "perplexity": perplexity,
        }

        if gold_labels is not None:
            cross_entropy = self.cross_entropy(cosine_similarity_scores, gold_labels)
            cross_perplexity = self.perplexity(cross_entropy)
            loglikelihood = self.loglikelihood(cross_entropy)
            metrics.update({
                "cross_entropy": cross_entropy,
                "cross_perplexity": cross_perplexity,
                "loglikelihood": loglikelihood
            })

        return metrics

class WavLMScorer(Scorer):
    pass