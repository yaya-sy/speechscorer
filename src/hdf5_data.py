"""Module for preparing input files for the models."""
from pathlib import Path
import logging
import h5py
import soundfile as sf
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def wavform(audio_path: Path):
    """Reads audio as waveform."""
    return sf.read(audio_path)

def h5_dataset(audio_folder: Path, output_folder: Path):
    """Creates h5py data for the dataloader."""
    output_folder.mkdir(exist_ok=True, parents=True)
    f = h5py.File(output_folder / "audio_arrays.hdf5", 'w')
    already_opened = dict()
    sorted_utterances = []
    LOGGER.info("Creating h5 dataset...")
    audios = list(audio_folder.glob("*.wav"))
    for audio in tqdm(audios):
        utterance_id = audio.stem
        already_opened.clear()
        audio, _ = wavform(audio)
        sorted_utterances.append((audio.shape[0], utterance_id))
        f.create_dataset(utterance_id, data=audio)
    sorted_utterances = sorted(sorted_utterances)
    _, ids = zip(*sorted_utterances)
    with open(output_folder / "utterances.sorted", 'w') as sorted_paths_file:
        sorted_paths_file.write(f"\n".join(ids))