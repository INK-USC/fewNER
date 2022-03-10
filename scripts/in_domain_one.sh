#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --job-name=in_domain
#SBATCH --gres=gpu:1 
#SBATCH --exclude ink-ellie
#SBATCH --exclude ink-ron

DATASET=$1
SHOTS=$2
PROMPT=$3
TEMPLATE=$4
TRAIN_SEED=$5
SAMPLE_SEED=$6


SEEDED_SUFFIX="${SHOTS}_${TRAIN_SEED}"


python3 transformers_trainer.py \
  --dataset $DATASET \
  --data_dir dataset/$DATASET \
  --model_folder models/$DATASET/max_basic \
  --device cuda:0 \
  --percent_filename_suffix $SEEDED_SUFFIX \
  --prompt $PROMPT \
  --template $TEMPLATE \
  --num_epochs 50 \
  --seed $TRAIN_SEED