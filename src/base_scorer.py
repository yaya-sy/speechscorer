"""TODO"""
from typing import Union, Dict, Optional, Iterable, List
from abc import ABC, abstractmethod
from pathlib import Path
from torch import Tensor
import torch.nn.functional as F
import torch
import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class BaseScorer:
    """
    This is a class for scoring raw speech.\
    The main idea is to examine how hesitant the model is\
    to realize the objective on which it has been trained.
    """
    def __init__(self,
                 model_checkpoint: Union[Path, str],
                 use_cuda: bool=False):
        self.use_cuda = use_cuda
        self.load_model(model_checkpoint)
        self.metrics = ["entropy", "perplexity"]
        self.with_targets_metrics = ["entropy", "perplexity","cross_entropy",\
            "cross_perplexity", "loglikelihood"]
    
    @abstractmethod
    def load_model(self, model_checkpoint: Union[Path, str]) -> None:
        pass

    @abstractmethod
    def forward_model(self,
                      x: Tensor,
                      labels: Optional[Tensor]=None,
                      ) -> Dict[str, Tensor]:
        pass
    
    def label_probabilities(self, scores: Tensor) -> Tensor:
        """
        Normalizes the label scores in order to get the\
        probability distribution over the labels.

        Parameters
        ----------
        - scores: Tensor
            Tensor representing the scores for each token to belong to a label.
            Shape=[batch_size, sequence_length, vocab_size]
        
        Returns
        -------
        - Tensor:
            The probaility distribution over the labels.
        """
        return F.softmax(scores, dim=-1)
    
    def entropy(self, probabilities: Tensor) -> Tensor:
        """
        Computes the casual entropy of probability distributions.
        This metric tells us how much the model is hesitant in her predictions.
        However, the entropy will be low when the the model is confident in her predictions\
        even if he's wrong. Cross entropy/log-likelihood is a solution to this problem.
        
        Parameters
        ----------
        - probabilities: Tensor
            The probability distribution over ids.
        Returns
        -------
        - float:
            The entropy of the probability distributions, the uncertainty of the model.
        """
        return (-torch.sum(probabilities * torch.log(probabilities), dim=-1)).mean(-1)

    def cross_entropy(self,
                      scores: Tensor,
                      labels: Tensor,
                      reduction: str="mean") -> Tensor:
        """
        Computes the cross entropy.

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
        return F.cross_entropy(scores, labels, reduction=reduction)

    def loglikelihood(self,
                      cross_entropy: Union[float, Tensor]
                      ) -> Union[float, Tensor]:
        """
        Computes the (log-)likelihhod of the labels given the cross-entropy.
        The loglihood is computed by just taking the opposit of the cross-entropy.
        
        Parameters
        ----------
        - cross_entropy: float
            The loglikelihood of the audio labels.
        """
        return -cross_entropy
    
    def perplexity(self, entropy_value: Tensor) -> Tensor:
        """Computes the perplexity from the (cross-)entropy."""
        return torch.exp(entropy_value)
        
    def compute_metrics(self,
                        forward_output: Dict[str, Optional[Tensor]],
                        labels: Optional[Tensor]=None
                        ) -> Iterable[tuple]:
        """
        Computes metrics from the output of the forward.
        
        Parameters
        ----------
        - forward_output: dict
            Associating each output to its tensor.
            The outputs are [logits, loss, labels]
        
        Returns
        -------
        - Tuple:
            Each item of the tuple corresponding to a metric.
            If the labels are not given, the items of the tuple\
            correspond to (entropy, perplexity).
            If the labels are given, the items correspond to\
            (cross_entropy, cross_perplexity, entropy,\
                perplexity, loglikelihood).
        """
        logits, loss, labels = forward_output["logits"], forward_output["loss"], forward_output["labels"]
        if logits.ndim != 3:
            raise ValueError("The shape of the output logits has to be of the form [batch_size, seq_length, vocab_size]")
        probabilities = self.label_probabilities(logits)
        entropies =  self.entropy(probabilities)
        perplexities = self.perplexity(entropies)
        if labels is not None and loss is None:
            batch_size, seq_len, vocab_size = logits.shape
            cross_entropies = self.cross_entropy(logits.view(-1, vocab_size),
                                                 labels.view(-1),
                                                 reduction="none")
            cross_entropies = cross_entropies.view(batch_size, seq_len).mean(-1)
            cross_perplexities = self.perplexity(cross_entropies)
            loglikelihood = self.loglikelihood(cross_entropies)
            batch_results = zip(entropies.tolist(), perplexities.tolist(),\
                                cross_entropies.tolist(), cross_perplexities.tolist(),
                                loglikelihood.tolist())
            return [dict(zip(self.with_targets_metrics, results)) for results in batch_results]

        batch_results = list(zip(entropies.tolist(), perplexities.tolist()))
        return [dict(zip(self.metrics, results)) for results in batch_results]

    def scores(self,
               x: Tensor,
               labels: Optional[Tensor]=None) -> List[tuple]:
        """
        Retrieves all the metrics for a given input_wavform

        Parameters
        ----------
        - input_wavform:
            Tensor representing the audio we want to score.
        
        Returns
        -------
        - List of tuples:
            Each layer with its associated metrics (entropy, perplexity, etc.)
        """
        # LOGGER.info("Computing statistics...")
        outputs = self.forward_model(x=x, labels=labels)
        # computing all statstics
        return self.compute_metrics(outputs, labels=labels)