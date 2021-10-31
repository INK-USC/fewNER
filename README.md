# fewNER

<!-- TODO: add description/abstract -->

## Table of contents

1. [Setup](#setup)
2. [Valid Combination Table](#valid-combination-table)
3. [Running](#running)

<hr/>

## Setup

1. _*Optional*_ Create and activate your conda/virtual environment

2. Run `pip install -r requirements.txt`

3. _*Optional*_ Add support for CUDA. We have tested the repository on pytorch version [1.7.1](https://pytorch.org/get-started/previous-versions/#v171) with CUDA version 10.1.

```bash
# conda
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

# pip
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

4. **Important** Locate your python libraries directory and replace the `bert_score/score.py` with `score.py` provided in this repository. We make some changes to cache the model and avoid reloading of model for each call. For example,

```bash
cp score.py ~/.conda/envs/<ENV_NAME>/lib/python3.6/site-packages/bert_score/score.py
```

<hr/>

## Valid Combination Table

| Prompt      | Template                                                         |
| ----------- | ---------------------------------------------------------------- |
| `max`       | `no_context`, `basic`, `basic_all`, `structure`, `structure_all` |
| `random`    | `no_context`, `basic`, `basic_all`, `structure`, `structure_all` |
| `sbert`     | `basic_all`, `structure_all`                                     |
| `bertscore` | `basic_all`, `structure_all`                                     |
| `search`    | `no_context`, `basic`, `basic_all`, `structure`, `structure_all` |

<hr/>

## Running

Possible values for:

- `<DATASET>` : `conll`, `ontonotes_conll`, `bc5cdr`
- `<PROMPT>` : from the table above
- `<TEMPLATE>` : from the table above

### Single run

- In-domain setting

  ```bash
  CUDA_VISIBLE_DEVICES=0 \
    python3 transformers_trainer.py \
    --dataset <DATASET> \
    --data_dir dataset/<DATASET> \
    --model_folder models/<DATASET>/test10 \
    --device cuda:0 \
    --percent_filename_suffix 50 \
    --prompt <PROMPT> \
    --template <TEMPLATE> \
    --num_epochs 50
  ```

- Domain Adaptation setting
  ```bash
  CUDA_VISIBLE_DEVICES=0 \
    python3 transformers_continual_trainer.py \
    --dataset <DATASET> \
    --data_dir dataset/<DATASET> \
    --checkpoint /home/shared/fewner/conll_all \
    --model_folder models/<DATASET>/transfer_train_50_conll \
    --device cuda:0 \
    --prompt <PROMPT> \
    --template <TEMPLATE> \
    --search_pool target \
    --percent_filename_suffix 50
  ```

### Multiple runs

- In-domain setting

  ```bash
  python sampling_run.py \
    --train_file transformers_trainer.py \
    --dataset <DATASET> \
    --data_dir dataset/<DATASET> \
    --gpu 0 \
    --suffix 50 \
    --prompt <PROMPT> \
    --template <TEMPLATE>
  ```

- Domain Adaptation setting
  ```bash
  python sampling_continual_run_slurm.py \
    --train_file transformers_continual_trainer.py \
    --dataset <DATASET> \
    --data_dir dataset/<DATASET> \
    --suffix 25 \
    --prompt <PROMPT> \
    --template <TEMPLATE> \
    --search_pool target \
    --checkpoint /home/shared/fewner/conll_all
  ```

### bash scripts

- To test your setup, `cd` to the repo root folder and run `./bin/test_setup`. This runs a training script and if you setup is correct you shouldn't see any errors.
- To train with conll dataset with a specific percent_filename_suffix, run `./bin/train_with_conll SUFFIX`. For example `./bin/train_with_conll 50`, or `./bin/train_with_conll all`.
- To clear **all** current models, results, and metrics to revert to a repo with just source code, run `./bin/clear_existing_outputs`. Note that by running this you will lose all the trained models and calculated results.
