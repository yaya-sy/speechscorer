"""Module implementing the HuBERT masked language model based scorer."""

from typing import Union, List, Optional, Dict
from .base_mlm_scorer import BaseMLMScorer
from pathlib import Path
import logging

from torch import Tensor
import torch
import fairseq

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class HubertMLMScorer(BaseMLMScorer):
    """
    HuBERT scorer using masked language model objective.
    """
    def __init__(self, model_checkpoint, use_cuda: bool=False):
        super().__init__(model_checkpoint=model_checkpoint, use_cuda=use_cuda)

    def load_model(self, model_checkpoint: Union[Path, str]) -> None:
        LOGGER.info(f"Loading model from {model_checkpoint}...")
        models, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_checkpoint])
        self.model = models[0].eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        LOGGER.info(f"Using device {self.device}")
        self.model.to(self.device)
        self.cfg = cfg

    def forward_model(self,
                      x: Tensor,
                      labels: Optional[Tensor],
                      batch_size: int=32
                      ) -> Dict[str, Tensor]:
        """Forward the input in order to get the logits."""
        with torch.no_grad() :
            x = x.to(self.device)
            # forward the convolution module in order to extract the features on the wavform
            features = self.model.feature_extractor(x)
            # apply HuBERT forward of these features
            features = features.transpose(1, 2)
            # features = hubert.layer_norm(features) # : Maybe not deactivate this even in eval mode
            features = self.model.post_extract_proj(features)
            ## Repeating the sequence so we can apply the masking on each token of the sequence.
            features = features.unsqueeze(1).repeat(1, features.shape[1], 1, 1) # shape[seq_len, seq_len, embedding_dim]
            # here, we create the boolean masking matrix with the value 'True' in the diagonal.
            masker = torch.zeros((features.shape[1], features.shape[1]))
            masker.fill_diagonal_(1)
            masker = masker.bool()
            # apply mask to each token in the sequence.
            features[:, masker] = self.model.mask_emb
            # run the BERT encoder
            contextual_vectors = []
            # run the BERT encoder
            for input_batch in features.view(-1, features.shape[2], features.shape[3]).split(batch_size):
                c, _ = self.model.encoder(input_batch)
                contextual_vectors.append(c)
            c = torch.cat(contextual_vectors)
            c = c.view(features.shape)
            # get only the masked tokens
            c = c[:, masker]
            # project the hidden vectors
            projected_c = self.model.final_proj(c)
            logits = self.cosine_similarity_scores(projected_hidden_vectors=projected_c,
                                                   temperature=self.cfg.model.logit_temp,
                                                   label_embeddings=self.model.label_embs_concat)
            return {"logits": logits.cpu(), "loss": None, "labels": labels}

    def scores(self,
               input_wavform: Tensor,
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
        outputs = self.forward_model(x=input_wavform, labels=labels)
        # computing all statstics
        return self.compute_metrics(outputs, labels=labels)