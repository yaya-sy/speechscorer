"""This module contains script for loading the data."""

from typing import Union, Iterator, Tuple, List, Iterable
from numbers import Number
from pathlib import Path
from itertools import islice, chain
import logging

import torch
from transformers import AutoProcessor
import numpy as np
import soundfile as sf

Frame = Tuple[Number, Number]
UtteranceId = str
Array = Union[np.array, torch.Tensor]
Batch = Iterable[Array]
Item = Tuple[Array, UtteranceId]
BatchItem = Tuple[Batch, List[UtteranceId]] 

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class DataLoader:
    """
    A dataloader that take as input a file containing\
    for each line an audio path, the onset and the offset\
    of the utterance. If the onsets and offsets are not specified,\
    the whole audio given in the audio path is considered\
    as one utterance.

    Parameters
    ----------
    - audios_folder: str, Path
        Folder containing the audios to score.
    - checkpoint: str
        The path to the huggingface checkpoint\
        of the processor.
    """

    def __init__(self,
                 input_audios: Union[List[Path], Path],
                 checkpoint
                 ) -> None:
        if not isinstance(input_audios, list):
            self.audios_files = list(input_audios.glob("*"))
        else:
            self.audios_files = input_audios
        self.sample_size = len(self.audios_files)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
            
    def audio_waveform(self, audio_path: Path) -> Array:
        """Reads an audio and extracts waveform."""
        return sf.read(audio_path)

    def data_iterator(self) -> Iterator[Tuple[str, str]]:
        """An iterator over audio frames and their corresponding labels (optional)."""
        for audio_file in self.audios_files:
            audio, sr = self.audio_waveform(audio_file)
            self.sampling_rate = sr
            yield audio_file.stem, audio

    def process_audio(self,
                      utterances: Union[Batch, Array],
                      padding,
                      max_length,
                      ) -> torch.Tensor:
        """Processes an audio."""
        inputs = self.processor(utterances,
                                return_tensors="pt",
                                sampling_rate=self.sampling_rate,
                                padding=padding,
                                truncation=False,
                                max_length=max_length)
        return inputs["input_values"] if "input_values" in inputs else inputs["input_features"]

    def __call__(self,
                 padding="longest",
                 max_length: int=20_000,
                 batch_size: int=32,
                 ) -> Iterator[BatchItem]:
        """Creates an iterator over batches."""
        iterator = self.data_iterator()
        for first in iterator:
            utterance_ids, audios = zip(*chain([first], islice(iterator, batch_size - 1)))
            batch_audios = self.process_audio(audios, padding=padding, max_length=max_length)
            yield batch_audios, list(utterance_ids)