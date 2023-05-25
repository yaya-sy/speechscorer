"""Import, load, and run the right scorer."""
from typing import Dict, Tuple

from .base_scorer import BaseScorer
from .mlm.hubert_mlm_scorer import HubertMLMScorer
from .mlm.wavlm_mlm_scorer import WavLMScorer
from .clm.whisper_clm_scorer import WhisperCLMScorer
from ..data_loader import DataLoader

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
    "wav-clm": WhisperCLMScorer,
    "hubert-clm": WhisperCLMScorer
}

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def run_scorer(scorer: BaseScorer,
               dataloader: DataLoader,
               batch_size: int=32
               ) -> Iterable[tuple]:
    """Runs the score on the data."""
    LOGGER.info("Scorer running...")
    total = dataloader.sample_size
    bar = tqdm(total=total)
    for x, utterance_ids in dataloader(batch_size):
        results = scorer.scores(x=x)
        assert (len(results) == len(utterance_ids),
                "Mismatch between utterances and their metrics.")
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
    parser.add_argument("-i", "--input",
                        help="""The input to score.\
                            Can be a folder of audio files or an audio file.""",
                        required=True)
    parser.add_argument("-m", "--model_checkpoint",
                        help="The path to the model checkpoint (huggingface or local).",
                        required=True)
    parser.add_argument("-s", "--scorer",
                        help="The scorer to use.",
                        choices=SCORERES.keys(),
                        default="whisper-clm")
    parser.add_argument("-b", "--batch_size",
                        help="The maximum of examples to feed to the model.",
                        type=int,
                        default=4)
    parser.add_argument('--use_cuda',
                        action="store_true",
                        default=False,
                        help="If --use_cuda and gpu is available, will use gpu.")
    args = parser.parse_args()
    output_folder = Path("../results")
    output_folder.mkdir(exist_ok=True, parents=True)
    scorer = init_scorer(args.scorer, args.model_checkpoint, args.use_cuda)
    processor_checkpoint = args.model_checkpoint if args.processor_checkpoint is None else args.processor_checkpoint
    dataloader = DataLoader(args.h5_data, args.utterances_file, processor_checkpoint)
    results = run_scorer(scorer, dataloader, batch_size=args.batch_size)
    df = pd.DataFrame(results)
    df.to_csv(output_folder / f"{args.output_filename}.csv")

if __name__ == "__main__":
    main()