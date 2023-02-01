"""This module contains script for loading the data."""
from typing import Union, Iterator, Tuple, Iterable, Dict, List, Optional
from numbers import Number
from pathlib import Path
from collections import defaultdict
from itertools import zip_longest, islice, chain
import torch
from transformers import AutoProcessor
import numpy as np
import soundfile as sf

Frame = Tuple[Number, Number]
DataPath = Union[str, Path]
Array = Union[np.array, torch.Tensor]
Labels = Union[list, List[None]]
DataItem = Dict[str, Union[list, Array]]
Batch = Dict[str, torch.Tensor]

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
        For each line, the first column is the audio path,\
        the second column is the onset of the utterance and the\
        last column is the offset of the utterance.
    - labels: Optional, str, Path.
        Path to the file containing labels for each utterance.\
        If given, this file must have the same lines as the utterances\
        file.
    - checkpoint: str
        The path to the huggingface checkpoint\
        of the processor.
    - pad_value: int
        The value to fill for padding.
    """

    def __init__(self,
                 utterances_path: DataPath,
                 labels_path: DataPath,
                 checkpoint: str,
                 pad_value: int=-100) -> None:
        self.utterances_path = utterances_path
        self.labels_path = labels_path
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.pad_value = pad_value
        self.sample_size = 0
        self.sampling_rate = None

    def audio_waveform(self, audio_path: DataPath) -> Array:
        """Reads an audio and extracts waveform."""
        return sf.read(audio_path)

    def load_utterances_paths(self, utterances_file: DataPath):
        """Loads file containing audio paths and (optional) the onset/offset."""
        utterances_data = defaultdict(list)
        with open(utterances_file, "r") as utterances:
            for utterance in utterances:
                utterance = utterance.strip()
                utterance_data = utterance.split("\t")
                n_data = len(utterance_data)
                if n_data in {3, 1}:
                    audio_path, onset, offset = utterance_data if n_data == 3 else (utterance, None, None)
                    onset, offset = (onset, offset) if None in {onset, offset} else (int(onset) / 1000, int(offset / 1000))
                else:
                    raise ValueError(f"The nummber of columns is unexpected.\
                        Expected 1 or 3 columns, get {len(utterance_data)}.")
                utterances_data[audio_path].append((onset, offset))
                self.sample_size += 1
        print(f"{self.sample_size} utterances loaded!")
        return dict(utterances_data)

    def extract_frames(self,
                       audio: Array,
                       frames: Iterable[Frame],
                       sr: int=16000) -> None:
        """Extracts frames of an audio waveform."""
        for onset, offset in frames:
            # if onset and offset are not specified,\
            # then the whole audio is considered as one utterance.
            if onset == offset == None:
                yield audio
                continue
            if onset > offset:
                raise ValueError(f"The onset ({onset}) is higher than the offset ({offset}).")
            onset *= sr
            offset *= sr
            if not (onset < audio.shape[0] >= offset):
                raise ValueError(f"The onset ({onset}) or the offset ({offset})\
                    are out of range for the waveform of length {audio.shape[0]}.")
            yield audio[onset:offset]

    def labels_iterator(self, labels_path: DataPath):
        """Reads and create an iterator from a given labels file."""
        with open(labels_path, "r") as utterances_labels:
            for labels in utterances_labels:
                labels.strip()
                yield [int(label) for label in labels.split(" ")]

    def audios_iterator(self, audio_paths: Dict):
        """An iterator over audio frames of an audio."""
        for audio_path in audio_paths:
            audio_waveform, sr = self.audio_waveform(audio_path)
            if self.sampling_rate is not None and sr != self.sampling_rate:
                raise ValueError("Inconsistent sampling rate across audios. Be sure all audio are sampled to the same rate.")
            else:
                self.sampling_rate = sr
            for audio_frame in self.extract_frames(audio_waveform,
                                            audio_paths[audio_path],
                                            sr):
                yield audio_frame

    def data_iterator(self) -> Iterator[DataItem]:
        """An iterator over audio frames and their corresponding labels (optional)."""
        audio_paths = self.load_utterances_paths(self.utterances_path)
        utterances = self.audios_iterator(audio_paths)
        all_labels = self.labels_iterator(self.labels_path) if self.labels_path is not None else [None]
        for utterance, labels in zip_longest(utterances, all_labels):
            yield utterance, labels
    
    def pad(self, batch: Batch):
        """Pad a given batch by filling a given pad value."""
        max_length = max(len(example) for example in batch)
        padded = []
        for example in batch:
            example += ([self.pad_value] * (max_length - len(example)))
            padded.append(example)
        return padded

    def __call__(self,
                 batch_size: int=32,
                 max_length: Optional[int]=None
                 ) -> Iterator[Batch]:
        """Creates i iterator over batches"""
        iterator = self.data_iterator()
        truncation = True if max_length is not None else False
        for first in iterator:
            utterances, labels = zip(*chain([first], islice(iterator, batch_size - 1)))
            inputs = self.processor(utterances,
                                    sampling_rate=self.sampling_rate,
                                    return_tensors="pt",
                                    padding=True,
                                    max_length=max_length,
                                    truncation=truncation)
            labels = self.pad(labels)
            inputs["labels"] = torch.tensor(labels) if None not in labels else None
            yield inputs
dl = DataLoader("utterances_file.txt", "labels_file.txt", "patrickvonplaten/wavlm-libri-clean-100h-base-plus")
for batch in dl(32):
    print(batch["input_values"].shape)
    print(batch["labels"].shape)