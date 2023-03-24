"""This module contains script for loading the data."""

from typing import Union, Iterator, Tuple, Dict, List, Optional
from numbers import Number
from pathlib import Path
from itertools import islice, chain
import logging

import torch
from transformers import AutoProcessor
import numpy as np
import soundfile as sf
import h5py
from tqdm import tqdm

Frame = Tuple[Number, Number]
DataPath = Union[str, Path]
Array = Union[np.array, torch.Tensor]
DataItem = tuple
Batch = Dict[str, torch.Tensor]

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
    - utterances: str, Path
        Path to the file storing the paths of utterances.\
        For each line, the first column is the utterance's id\
        the second column contains the audio path,\
        the third column is the onset of the utterance and the\
        last column is the offset of the utterance.
    - targets: Optional, str, Path.
        Path to the file containing targets (entropies) for each utterance.\
        If given, this file must have the same lines as the utterances\
        file.
    - checkpoint: str
        The path to the huggingface checkpoint\
        of the processor.
    """

    def __init__(self,
                 h5_file: DataPath,
                 sorted_utterances_file: DataPath,
                 checkpoint: str,
                 sampling_rate: int=16000,
                 ) -> None:
        self.h5_file = h5py.File(h5_file)
        self.sampling_rate = sampling_rate
        with open(sorted_utterances_file, "r") as sorted_utterances:
            self.ids = [line.strip() for line in sorted_utterances]
        self.sample_size = len(self.ids)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.has_timemarks = None
            
    def audio_waveform(self, audio_path: DataPath) -> Array:
        """Reads an audio and extracts waveform."""
        return sf.read(audio_path)

    def audios_iterator(self):
        """An iterator over audio frames of an audio."""
        for utterance_id in self.ids:
            yield utterance_id, self.h5_file[utterance_id][:]

    def data_iterator(self) -> Iterator[DataItem]:
        """An iterator over audio frames and their corresponding labels (optional)."""
        for utterance_id, utterance in self.audios_iterator():
            yield utterance_id, utterance

    def __call__(self,
                 batch_size: int=32
                 ) -> Iterator[Batch]:
        """Creates an iterator over batches."""
        iterator = self.data_iterator()
        for first in iterator:
            utterance_ids, utterances = zip(*chain([first], islice(iterator, batch_size - 1)))
            inputs = self.processor(utterances,
                                    sampling_rate=self.sampling_rate,
                                    return_tensors="pt")
            x = inputs["input_values"] if "input_values" in inputs else inputs["input_features"]
            yield x, list(utterance_ids)