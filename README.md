# fewNER


## Table of contents

1. [Setup](#setup)
2. [Valid Combination Table](#valid-combination-table)
3. [Running](#running)

   3.1. [Single run](#single-run)

   3.2. [Multiple runs](#multiple-runs)

   3.3. [Running prompt Search](#running-prompt-search)

<hr/>

## Setup

1. _*Optional*_ Create and activate your conda/virtual environment

2. Run `pip install -r requirements.txt`

3. _*Optional*_ Add support for CUDA. We have tested the repository on pytorch
   version [1.7.1](https://pytorch.org/get-started/previous-versions/#v171) with CUDA version 10.1.

```bash
# conda
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

# pip
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

4. **Important** Locate your python libraries directory and replace the `bert_score/score.py` with `score.py` provided
   in this repository. We make some changes to cache the model and avoid reloading of model for each call. For example,

```bash
cp score.py ~/.conda/envs/<ENV_NAME>/lib/python3.6/site-packages/bert_score/score.py
```

<hr/>

## Valid Combination Table

| Prompt      | Template                                                         |
| ----------- | ---------------------------------------------------------------- |
| `max`       | `no_context`, `basic`, `lexical` |
| `random`    | `no_context`, `basic`, `lexical`|
| `sbert`     | `basic_all`, `lexical_all`                                     |
| `bertscore` | `basic_all`, `lexical_all`                                     |

<hr/>

## Running

Possible values for:

- `<DATASET>` : `conll`, `ontonotes_conll`, `bc5cdr`
- `<PROMPT>` : from the table above
- `<TEMPLATE>` : from the table above
- `<SUFFIX>` : 25, 50
- `<TRAIN_SEED>` : 42, 1337, 2021
- `<SAMPLE_SEED>` : 42, 1337, 2021, 5555, 9999
- `<CHECK_POINT>` : Saved checkpoint

### Single run

Execute a single run.

- In-domain setting

  ```bash
  scripts/in_domain/in_domain_one.sh <DATASET> <SHOT> <PROMPT> <TEMPLATE> <TRAIN_SEED> <SAMPLE_SEED>
  ```

- Domain Adaptation setting
  ```bash
  scripts/domain_adaptation/domain_adaptation_one.sh <DATASET> <SHOT> <PROMPT> <TEMPLATE> <TRAIN_SEED> <SAMPLE_SEED> <CHECK_POINT>
  ```

### Multiple runs

This setting runs all 15 runs i.e. 5 different sub-samples x 3 training seeds

- In-domain setting

  ```bash
  scripts/in_domain/in_domain_all.sh
  ```
  * remember to configure the parameters on top of this script.

- Domain Adaptation setting
  ```bash
  scripts/domain_adaptation/domain_adaptation_all.sh
  ```

### Running prompt Search

| Prompt      | Template                                                         |
| ----------- | ---------------------------------------------------------------- |
| `search`    | `no_context`, `basic`, `lexical` |

1. search for best entities (based on only one seed)
    ```bash
    python3 search.py \
        --dataset <DATASET> \
        --data_dir dataset/<DATASET> \
        --model_folder models/<DATASET>/conll_max_basic \
        --device cuda:0 \
        --percent_filename_suffix <SEEDED_SUFFIX> \
        --template <TEMPLATE>
    ```

2. Run with best entities
    ```bash
    python sampling_run.py \
        --train_file search_run.py \
        --dataset <DATASET> \
        --data_dir dataset/<DATASET> \
        --gpu 0 \
        --suffix <SUFFIX> \
        --template <TEMPLATE>
    ```