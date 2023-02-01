from masked_language_modeling.hubert_mlm_scorer import HubertMLMScorer

from argparse import ArgumentParser
from typing import Iterable
from data_loader import data_iterator
from fairseq.data.audio.audio_utils import get_features_or_waveform
import torch

def run_scorer(scorer, data) -> Iterable[tuple]:
    use_target = False
    columns = None
    results = []
    for audio_path, wavform, labels in data:
        if columns is None:
            if labels is None:
                columns = ["layer", "entropy", "perplexity"]
            if labels is not None:
                use_target = True
                columns = ["layer", "cross_entropy", "entropy", "cross_perplexity", "perplexity", "log_likelihood"]
        x = torch.from_numpy(wavform).float().unsqueeze(0)
        y = torch.tensor(labels) if use_target else None
        metrics = scorer.scores(input_wavform=x, gold_labels=y)
        metrics["audio_path"] = audio_path
        results.append(metrics)
    return results, columns

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_audios",
                        help="The .tsv file containing the audio paths in the first column\
                             and (optional) the labels of the audios in the second column.")
    parser.add_argument("-s", "--scorer",
                        help="The scorer to use.")
    parser.add_argument("-o", "--output_folder",
                        help="The .tsv file containing the audio paths in the first column\
                             and (optional) the labels of the audios in the second column.")
    args = parser.parse_args()
    scorer = "TODO"
    data = data_iterator(args.input_audios)
    results, columns = run_scorer(scorer, data)

if __name__ == "__main__":
    pass
path = "/scratch2/ysy/DATA/PROVIDENCE/wnh/audios/Alex/010427/Mother/utterance_0214.wav" # ɡ_ʊ_d_n_aɪ_n_d-en-US-Wavenet-J
# /scratch2/whavard/DATA/LSFER/providence/recordings/raw/Alex/Alex_030516.wav
scorer = HubertMLMScorer("checkpoints/hubert_base_ls960.pt")
wavform = get_features_or_waveform(path, need_waveform=True, use_sample_rate=16000)
x = torch.from_numpy(wavform).float().unsqueeze(0)
y = torch.randint(4, 504, (123, ))
print(x.shape)
metrics = scorer.scores(input_wavform=x, labels=None)
print(list(metrics))