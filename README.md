# speechscorer: a simple unsupervised spoken utterance scorer.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yaya-sy/speechscorer/blob/main/demo/speechscorer.ipynb)

<p align="center">
    <img width="580" alt="speechocean correlations" src="plots/hubert-mlm-scorer.png">
</p>

This package enables you to assign a score to a given spoken utterance (audio). This score can reflect the intelligbility or the fluency/grammaticality of the utterance.

Different models pretrained with different training objectives are proposed: HuBERT, Whisper and WavLM.

# How it works
The scoring method relies on the internal model hesitation (entropy) while predicting for the input speech. Depending on the model and its training objective, the entropy is computed differently. You can learn more about how it works by reading the documentation:
- [Entropy of models trained with the masked language model objective](speechscorer/mlm/README.md)
- [Entropy of models trained with the ASR objective](speechscorer/clm/README.md)

# Getting start

You will first need to istall the right [PyTorch](https://pytorch.org/get-started/locally/) for your computer

Then you can install speechscorer using pip:
```bash
pip install git+https://github.com/yaya-sy/speechscorer.git
```

once installed, you can score your utterance with this command:

```bash
speechscore -a <your-audio>
```

To see more available options, run:
```bash
speechscore -h
```
# Demo

You can find a colab notebook in `demo/speechscorer.ipynb` for an example of use case.

# Biases
The models are trained using speech from dominant populations: Western, affluent and white man ways of speaking English. Model results are therefore strongly influenced by the way English is spoken by these dominant populations. Scores can be low for native English but from under-represented populations (Jamaican English, Nigerian English, etc.) and from women.

# Citation

To refenrece speechscorer in your own work, please add the following citation.

```bibtex
@article{SY2023,
  url = {https://github.com/yaya-sy/speechscorer},
  year = {2023},
  author = {Yaya SY},
  title = {speechscorer: A simple unsupervised spoken utterances scorer},
  journal = {Journal of Open Source Software}
}
