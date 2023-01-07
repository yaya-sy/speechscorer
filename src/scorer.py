"""Module implementing the different type of scorers"""
from abc import ABC, abstractmethod
from typing import Union, Set, Dict
from pathlib import Path

Tensor = "torch.Tensor"

class BaseScorer(ABC):
    """TODO"""
    def __init__(self, checkpoint: Union[Path, str], logits_functions: Set[str]):
        self.accepted_logits_functions: Dict[str] = {
            "similarity": self.contrastive,
            "normalized_label_projection": self.normalized_projection,
            "projection_label_projection": self.projection
            }
        self.check_logits_functions(logits_functions)
        self.model = self.load_model(checkpoint)
    
    @abstractmethod
    def load_model(self, checkpoint: Union[Path, str]) -> None:
        pass

    def check_logits_functions(self, logits_functions: Set[str]) -> None:
        if set(self.accepted_logits_functions.keys()) != logits_functions:
            not_accepted_logits = logits_functions.difference(self.accepted_logits_functions)
            print(f"{not_accepted_logits} are not in the proposed logits functions. They will be ignored.")
    
    def similarity(self,
                   projected_hidden_vectors: Tensor,
                   label_embeddings: Tensor) -> Tensor:
        """TODO"""

        return Tensor
    
    def normalized_label_projection(self,
                                    projected_hidden_vectors: Tensor,
                                    label_embeddings: Tensor) -> Tensor:
        """TODO"""
        return Tensor
    
    def label_projection(self,
                         projected_hidden_vectors: Tensor,
                         label_embeddings: Tensor) -> Tensor:
        """TODO"""
        return Tensor
    
    def label_probabilities(self, logits) -> Tensor:
        """TODO"""
        return Tensor
    
    def entropy(self, probabilities) -> Tensor:
        """TODO"""
        return Tensor
    
    def cross_entropy(self, logits) -> float:
        """TODO"""
        return float
    
    def perplexity(self, entropy_value: float) -> float:
        """TODO"""
        return float