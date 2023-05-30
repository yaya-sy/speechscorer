"""Import, load, and run the right scorer."""
from .base_scorer import BaseScorer
from .mlm.hubert_mlm_scorer import HubertMLMScorer
from .mlm.wavlm_mlm_scorer import WavLMScorer
from .clm.whisper_clm_scorer import WhisperConditionalLanguageModelScorer
from .clm.wavlm_clm_scorer import WavLMConditionalLanguageModelScorer
from .clm.hubert_clm_scorer import HuBERTConditionalLanguageModelScorer
from .data_loader import DataLoader

from typing import Dict, Tuple

from argparse import ArgumentParser
from typing import Iterable
import logging
from pathlib import Path

from tqdm import tqdm
import pandas as pd

SCORERES: Dict[str, Tuple[str, BaseScorer]] = {
    "hubert-mlm": HubertMLMScorer,
    "wavlm-mlm": WavLMScorer,
    "whisper-clm": WhisperConditionalLanguageModelScorer,
    "wavlm-clm": WavLMConditionalLanguageModelScorer,
    "hubert-clm": HuBERTConditionalLanguageModelScorer
}

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def run_scorer(scorer: BaseScorer,
               dataloader: DataLoader,
               padding="max_length",
               max_length: int=None,
               batch_size: int=4
               ) -> Iterable[tuple]:
    """Runs the score on the data."""
    LOGGER.info("Scorer running...")
    total = dataloader.sample_size
    bar = tqdm(total=total)
    for x, utterance_ids in dataloader(padding=padding, max_length=max_length, batch_size=batch_size):
        results = scorer.scores(x=x)
        assert len(results) == len(utterance_ids), "Mismatch between utterances and their metrics."
        for utterance_id, result in zip(utterance_ids, results):
            result["utterance_id"] = utterance_id
            yield result
        bar.update(x.shape[0])

def init_scorer(scorer_name: str,
                model_checkpoint,
                use_gpu: bool=False
                ) -> BaseScorer:
    """Initializes a scorer class."""
    scorer = SCORERES[scorer_name]
    scorer = scorer(model_checkpoint, use_gpu)
    return scorer

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-a", "--audio",
                        type=str,
                        help="""The input audio to score.\
                            Can be a folder of audio files or an audio file.""",
                        required=True)
    parser.add_argument("-m", "--model_checkpoint",
                        type=str,
                        help="The path to the model checkpoint (huggingface or local).",
                        default="openai/whisper-base.en",
                        required=False)
    parser.add_argument("-p", "--processor_checkpoint",
                        type=str,
                        help="The audio processor for the input audios.",
                        required=False)
    parser.add_argument("-s", "--scorer",
                        type=str,
                        help="The scorer to use.",
                        choices=SCORERES.keys(),
                        default="whisper-clm",
                        required=False)
    parser.add_argument("-b", "--batch_size",
                        type=int,
                        help="The maximum of examples to feed to the model.",
                        required=False,
                        default=1)
    parser.add_argument("-d", "--padding",
                        type=str,
                        help="The padding strategy",
                        required=False,
                        default="max_length")
    parser.add_argument("-l", "--max_length",
                        type=int,
                        help="The the max length of the input sequence to consider.",
                        required=False,
                        default=None)
    parser.add_argument('--use-gpu',
                        action="store_true",
                        default=False,
                        help="will use gpu if --use_cuda and gpu is available.")
    return parser.parse_args()

def main():
    args = get_args()
    output_folder = Path("results")
    output_folder.mkdir(exist_ok=True, parents=True)
    scorer = init_scorer(args.scorer, args.model_checkpoint, args.use_gpu)
    processor_checkpoint = args.model_checkpoint if args.processor_checkpoint is None else args.processor_checkpoint
    input_audios = Path(args.audio)
    if input_audios.is_file():
        input_audios = [input_audios]
    dataloader = DataLoader(input_audios, processor_checkpoint)
    results = run_scorer(scorer,
                         dataloader,
                         padding=args.padding,
                         max_length=args.max_length,
                         batch_size=args.batch_size)
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(output_folder / f"results.csv", index=None)

if __name__ == "__main__":
    main()