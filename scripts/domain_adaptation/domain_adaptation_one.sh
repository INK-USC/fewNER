#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --job-name=in_domain
#SBATCH --gres=gpu:1
#SBATCH --exclude ink-ellie,ink-molly,ink-ron

DATASET=$1
SHOTS=$2
PROMPT=$3
TEMPLATE=$4
TRAIN_SEED=$5
SAMPLE_SEED=$6
CHECK_POINT=$7

SEEDED_SUFFIX="${SHOTS}_${SAMPLE_SEED}"
MODEL_NAME="model_da_${PROMPT}_${TEMPLATE}_${SEEDED_SUFFIX}_${TRAIN_SEED}"

python3 transformers_continual_trainer.py \
  --dataset $DATASET \
  --data_dir dataset/$DATASET \
  --checkpoint $CHECK_POINT \
  --model_folder models/$DATASET/$MODEL_NAME \
  --device cuda:0 \
  --prompt $PROMPT \
  --template $TEMPLATE \
  --search_pool target \
  --percent_filename_suffix $SEEDED_SUFFIX \
  --seed $TRAIN_SEED
