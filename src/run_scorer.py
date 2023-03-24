"""Import, load, and run the right scorer."""
from typing import Dict, Tuple, Optional

from .base_scorer import BaseScorer
from .masked_language_modeling.hubert_mlm_scorer import HubertMLMScorer
from .masked_language_modeling.wavlm_mlm_scorer import WavLMScorer
from .conditional_language_modeling.whisper_clm_scorer import WhisperCLMScorer
from .attention_based_scoring.whisper_attentions_scorer import WhisperAttentionScorer
from .data_loader import DataLoader

from argparse import ArgumentParser
from typing import Iterable
import logging
from pathlib import Path

from tqdm import tqdm
import pandas as pd

SCORERES: Dict[str, Tuple[str, BaseScorer]] = {
    "hubert-mlm": HubertMLMScorer,
    "wavlm-mlm": WavLMScorer,
    "whisper-clm": WhisperCLMScorer,
    "whisper-attention": WhisperAttentionScorer
}

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def repeate_utterances(utterances, repeater):
    encoder, cross = [], []
    len_repeater = len(repeater)
    for utterance in utterances:
        cross.extend([utterance] * repeater[0])
        if len_repeater == 2:
            encoder.extend([utterance] * repeater[1])
    return cross + encoder

def run_scorer(scorer: BaseScorer,
               dataloader: DataLoader,
               batch_size: int=32
               ) -> Iterable[tuple]:
    """Run the score on the data."""
    LOGGER.info("Scorer running...")
    total = dataloader.sample_size
    bar = tqdm(total=total)
    for x, utterance_ids in dataloader(batch_size):
        results, *repeater = scorer.scores(x=x)
        if repeater:
            utterance_ids = repeate_utterances(utterance_ids, *repeater)
        assert len(results) == len(utterance_ids), "Mismatch between utterances and their metrics."
        for utterance_id, result in zip(utterance_ids, results):
            result["utterance_id"] = utterance_id
            yield result
        bar.update(x.shape[0])

def init_scorer(scorer_name: str,
                model_checkpoint,
                use_cuda: bool=False
                ) -> BaseScorer:
    """Initializes a scorer class."""
    scorer = SCORERES[scorer_name]
    scorer = scorer(model_checkpoint, use_cuda)
    return scorer

def main():
    parser = ArgumentParser()
    parser.add_argument("-u", "--utterances_file",
                        help="The file containing the utterance_ids.",
                        required=True)
    parser.add_argument("-d", "--h5_data",
                        help="The audio utterances as numpy array.",
                        type=str,
                        required=True)
    parser.add_argument("-l", "--labels_file",
                        help="(Optional) Path to the file containing labels for each utterance in the utterances file.",
                        required=False)
    parser.add_argument("-m", "--model_checkpoint",
                        help="The path to the model checkpoint (huggingface or local).",
                        required=True)
    parser.add_argument("-p", "--processor_checkpoint",
                        help="The path to the processor checkpoint (huggingface or local).",
                        required=False,
                        default=None)
    parser.add_argument("-s", "--scorer",
                        help="The scorer to use.",
                        choices=SCORERES.keys(),
                        default="whisper-clm")
    parser.add_argument("-b", "--batch_size",
                        help="The batch size.",
                        type=int,
                        default=32)
    parser.add_argument('--use_cuda',
                        action="store_true",
                        default=False,
                        help="If --use_cuda and gpu is available, will use gpu.")
    parser.add_argument("-o", "--output_folder",
                        help="The folder where the output will be stored.",
                        required=True)
    parser.add_argument("-n", "--output_filename",
                        help="The output filename.",
                        required=True)
    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    scorer = init_scorer(args.scorer, args.model_checkpoint, args.use_cuda)
    processor_checkpoint = args.model_checkpoint if args.processor_checkpoint is None else args.processor_checkpoint
    dataloader = DataLoader(args.h5_data, args.utterances_file, processor_checkpoint)
    results = run_scorer(scorer, dataloader, batch_size=args.batch_size)
    df = pd.DataFrame(results)
    df.to_csv(output_folder / f"{args.output_filename}.csv")

if __name__ == "__main__":
    main()