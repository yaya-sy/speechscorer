"""TODO"""
from typing import Union, Dict, Optional
from abc import ABC, abstractmethod
from pathlib import Path
from torch import Tensor
import torch.nn.functional as F
import torch

class BaseScorer(ABC):
    """
    This is a class for scoring raw speech.\
    The main idea is to examine how hesitant the model is\
    to realize the objective on which it has been trained.
    """
    def __init__(self,
                 path_to_checkpoint: Union[Path, str]):

        self.load_model(path_to_checkpoint)

    @abstractmethod
    def load_model(self, path_to_checkpoint: Union[Path, str]) -> None:
        pass

    @abstractmethod
    def forward_model(self,
                      input: Tensor,
                      labels: Optional[Tensor]=None,
                      batch_size: Optional[int]=32
                      ) -> Dict[str, Tensor]:
        pass
    
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
    
    def entropy(self, probabilities: Tensor) -> Tensor:
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
        return (-torch.sum(probabilities * torch.log(probabilities), dim=-1)).mean(-1)

    def loglikelihood(self, cross_entropy: Union[float, Tensor]) -> Union[float, Tensor]:
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
    
    def compute_metrics(self, forward_output: Dict[str, Optional[Tensor]]):
        logits, loss, labels = forward_output["logits"], forward_output["loss"], forward_output["labels"]
        probabilities = self.label_probabilities(logits)
        entropies =  self.entropy(probabilities)
        perplexities = self.perplexity(entropies)
        layers = range(1, len(entropies) + 1)

        if labels is not None and loss is None:
            n_layers, seq_length, _ = logits.shape
            gold_labels = gold_labels.unsqueeze(0).repeat(n_layers, 1)
            cross_entropies = self.cross_entropy(logits.view(n_layers * seq_length, -1),
                                                 gold_labels.view(-1),
                                                 reduction="none")
            cross_entropies = cross_entropies.view(n_layers, seq_length).mean(-1)
            cross_perplexities = self.perplexity(cross_entropies)
            loglikelihood = self.loglikelihood(cross_entropies)
            return(zip(layers, cross_entropies.tolist(), entropies.tolist(),
                        cross_perplexities.tolist(), perplexities.tolist(),
                        loglikelihood.tolist()))
        return zip(layers, entropies.tolist(), perplexities.tolist())