from scorer import HuBERTScorer
from fairseq.data.audio.audio_utils import get_features_or_waveform
import torch

path = "/scratch1/data/derived_data/lang_acq_model_evaluation/lexical/test/ɡ_ʊ_d_n_aɪ_n_d-en-US-Wavenet-J.wav" # ɡ_ʊ_d_n_aɪ_n_d-en-US-Wavenet-J
scorer = HuBERTScorer("checkpoints/hubert_base_ls960.pt")
wavform = get_features_or_waveform(path, need_waveform=True, use_sample_rate=16000)
x = torch.from_numpy(wavform).float().unsqueeze(0)
metrics = scorer.scores(x)
print(metrics)