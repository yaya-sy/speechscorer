"""This module contains script for loading the data."""
from typing import Union, Iterator
from pathlib import Path
import numpy as np
from fairseq.data.audio.audio_utils import get_features_or_waveform

def data_iterator(tsv_file: Union[Path, str],
                  sampling_rate: int
                  ) -> Iterator[str, str]:
    """An iterator over audio files and their labels."""
    with open(tsv_file, "r") as input_file:
        labels_path = None
        for line in input_file:
            line = line.strip()
            paths = input_file.split("\t")
            if len(paths) == 2:
                audio_path, labels_path = paths
            else:
                labels_path = paths[0]
            wavform = get_features_or_waveform(audio_path, need_waveform=True, use_sample_rate=sampling_rate)
            assert wavform.ndim == 1, f"Expected one dimension vector, given {wavform.ndim}"
            with open(labels_path) as labels_file:
                labels = next(labels_file).strip()
            yield wavform, labels

    
