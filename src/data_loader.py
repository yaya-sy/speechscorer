"""This module contains script for loading the data."""
from typing import Union, Iterator
from pathlib import Path
import numpy as np

def data_iterator(tsv_file: Union[Path, str],
                  format: str="wav") -> Iterator[str, str]:
    """An iterator over audio files and their labels."""
    with open(tsv_file, "r") as input_file:
        for line in tsv_file:
            line = line.strip()
            audio_path, labels_path = tsv_file.split("\t")
            with open(labels_path) as labels_file:
                labels = next(labels_file).strip()
            yield audio_path, labels

def audio_process(audio_path:Union[Path, str]) -> np.array:
    pass
    
