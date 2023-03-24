"""Module implementing abstract conditional language model based scorer."""
from typing import Dict, Optional, Iterable
from math import exp
from ..base_scorer import BaseScorer
from typing import Union
from pathlib import Path
from torch import Tensor
import torch

class BaseAttentionScorer(BaseScorer):
    """
    Base scorer for conditional language model speech models.\
    The logits are the scores over the vocabulary\
    at each step of the prediction. So computing the entropy on those\
    scores can be seen as the hesitation of the model on its predictions.
    """
    def __init__(self, model_checkpoint: Union[Path, str], use_cuda: bool = False):
        super().__init__(model_checkpoint, use_cuda)

    def dictify(self,
                attention_entropies: torch.Tensor,
                attention_type: str,
                ) -> Iterable[Dict]:
        """
        Makes dictionaries from attentions entropies.
        
        Parameters
        ---------
        - attention_entropies: Tensor
            entropies of attentions. Shape=[batch, layers, heads]
        """
        dicts = []
        for batch in attention_entropies.tolist():
            for layer, layer_heads in reversed(list(enumerate(batch, start=1))):
                for head, entropy in reversed(list(enumerate(layer_heads, start=1))):
                    dicts.append({
                        "layer": layer,
                        "head": head,
                        "entropy": entropy,
                        "perplexity": exp(entropy),
                        "attentions": attention_type
                    })
        return dicts
    
    def compute_metrics(self,
                        forward_output: Dict[str, Optional[Tensor]],
                        labels: Optional[Tensor] = None
                        ) -> Iterable[tuple]:
        
        """Given the forward outputs, compute the attentions entropies of the model."""
        with torch.no_grad():
            attentions = forward_output["cross_attentions"]
            attentions_entropy = self.entropy(attentions)
            _, layers, heads = attentions_entropy.shape
            repeater = [layers * heads]
            results = self.dictify(attentions_entropy, "cross")
            if forward_output["encoder_attentions"] is not None:
                attentions = forward_output["encoder_attentions"]
                attentions_entropy = self.entropy(attentions_entropy)
                _, layers, heads = attentions_entropy.shape
                repeater.append(layers * heads)
                results += self.dictify(attentions_entropy, "encoder")
            return results, repeater


