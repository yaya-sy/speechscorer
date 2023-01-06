"""This module contains script for loading the data."""

from pathlib import Path

def data_iterator(input_folder: Path, format=".wav"):
    """TODO"""
    for audio_file in input_folder.glob(f"*.{format}"):
        filename = audio_file.stem
        with open(f"{input_folder}/{filename}.labels") as label_file:
            labels = next(labels)
        audio = "TODO"
        yield audio, labels