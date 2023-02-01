"""Module implementing the HuBERT scorer using MLM obkective."""
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

    def __init__(self, path_to_checkpoint):
        super().__init__(path_to_checkpoint=path_to_checkpoint)
    
    def load_model(self, path_to_checkpoint: Union[Path, str]) -> None:
        LOGGER.info(f"Loading model from {path_to_checkpoint}...")
        models, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([path_to_checkpoint])
        self.model = models[0].eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LOGGER.info(f"Using device {self.device}")
        self.model.to(self.device)
        self.cfg = cfg

    def forward_model(self,
                     input: Tensor,
                     labels: Optional[Tensor]=None,
                     batch_size: int=32) -> Dict[str, Tensor]:
        """Forward the input in order to get the logits."""
        input = input.to(self.device)
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
            print(masker.shape)
            # apply mask to each token in the sequence.
            features[masker] = self.model.mask_emb
            layers_hidden_vectors = []
            # run the BERT encoder
            for input_batch in features.split(batch_size):
                _, layers = self.model.encoder(input_batch)
                h_layers = torch.stack([x.transpose(0, 1) for x, _, _ in layers])
                layers_hidden_vectors.append(h_layers)
            layers_hidden_vectors = torch.cat(layers_hidden_vectors, 1)
            # get only the masked tokens
            masked_layers_hidden_vectors = layers_hidden_vectors[:, masker]
            # project the hidden vectors
            return self.model.final_proj(masked_layers_hidden_vectors)

    def scores(self,
               input_wavform: Tensor,
               labels: Optional[Tensor]=None,
               batch_size=32) -> List[tuple]:
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
        LOGGER.info("Computing statistics...")
        projected_hidden_vectors = self.forward_model(input=input_wavform, labels=labels, batch_size=batch_size)
        cosine_similarity_scores = self.cosine_similarity_scores(projected_hidden_vectors=projected_hidden_vectors,
                                                                 temperature=self.cfg.model.logit_temp,
                                                                 label_embeddings=self.model.label_embs_concat)
        forward_output = {"logits": cosine_similarity_scores, "loss": None, "labels": labels}
        # computing all statstics
        return self.compute_metrics(forward_output)