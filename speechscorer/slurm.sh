#!/bin/bash
#SBATCH --cpus-per-task=14
#SBATCH --partition=all
#SBATCH --job-name=metrics
#SBATCH --time=30:00:00
#SBATCH --output=%x-%j.log

echo "Running job on $(hostname)"

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate sscorer

# lunch the testing
python -m src.run_scorer -u ../DATA/PROVIDENCE/wnh/h5/providence_utterances.txt -m openai/whisper-tiny.en -s whisper-attention -o results -b 32 -p openai/whisper-tiny.en -f ../DATA/PROVIDENCE/wnh/h5/providence.hdf5 -n whisper_attention_tiny_en