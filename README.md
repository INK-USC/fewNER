# fewNER

### Setup
1. Locate your python libraries directory and replace the `~/.conda/envs/<ENV_NAME>/lib/python3.6/site-packages/bert_score/score.py` with the [`./score.py`](./score.py) provided in this repositories root

### bash scripts
* To test your setup, `cd` to the repo root folder and run `./bin/test_setup`. This runs a training script and if you setup is correct you shouldn't see any errors. 
* To train with conll dataset with a specific percent_filename_suffix, run `./bin/train_with_conll SUFFIX`. For example `./bin/train_with_conll 50`, or `./bin/train_with_conll all`.
* To clear **all** current models, results, and metrics to revert to a repo with just source code, run `./bin/clear_existing_outputs`. Note that by running this you will lose all the trained models and calculated results. 


### Commands Memo:
```
python sampling_run.py --train_file transformers_trainer.py --dataset conll --data_dir dataset/conll --gpu 0 --suffix 50
python sampling_run.py --train_file transformers_trainer.py --dataset conll --data_dir dataset/conll --gpu 1 --suffix 100
python sampling_run.py --train_file transformers_trainer.py --dataset conll --data_dir dataset/conll --gpu 2 --suffix 150
python sampling_run.py --train_file transformers_trainer.py --dataset conll --data_dir dataset/conll --gpu 3 --suffix 200

python sampling_run.py --train_file transformers_trainer.py --dataset bc5cdr --data_dir dataset/bc5cdr --gpu 0 --suffix 50
python sampling_run.py --train_file transformers_trainer.py --dataset bc5cdr --data_dir dataset/bc5cdr --gpu 1 --suffix 100
python sampling_run.py --train_file transformers_trainer.py --dataset bc5cdr --data_dir dataset/bc5cdr --gpu 2 --suffix 150
python sampling_run.py --train_file transformers_trainer.py --dataset bc5cdr --data_dir dataset/bc5cdr --gpu 3 --suffix 200

python sampling_run.py --train_file transformers_trainer.py --dataset ontonotes_conll --data_dir dataset/ontonotes_conll --gpu 0 --suffix 50
python sampling_run.py --train_file transformers_trainer.py --dataset ontonotes_conll --data_dir dataset/ontonotes_conll --gpu 1 --suffix 100
python sampling_run.py --train_file transformers_trainer.py --dataset ontonotes_conll --data_dir dataset/ontonotes_conll --gpu 2 --suffix 150
python sampling_run.py --train_file transformers_trainer.py --dataset ontonotes_conll --data_dir dataset/ontonotes_conll --gpu 3 --suffix 200
```

```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_all \
--device cuda:0 \
--percent_filename_suffix all
```

```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_bertscore \
--device cuda:0 \
--percent_filename_suffix 50 \
--prompt bertscore
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_100_sbert \
--device cuda:0 \
--percent_filename_suffix 100 \
--prompt sbert
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_150_sbert \
--device cuda:0 \
--percent_filename_suffix 150 \
--prompt sbert
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_200_sbert \
--device cuda:0 \
--percent_filename_suffix 200 \
--prompt sbert
```

```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_50_sbert \
--device cuda:0 \
--percent_filename_suffix 50 \
--prompt sbert
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_100_sbert \
--device cuda:0 \
--percent_filename_suffix 100 \
--prompt sbert
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_150_sbert \
--device cuda:0 \
--percent_filename_suffix 150 \
--prompt sbert
```

```
CUDA_VISIBLE_DEVICES=0 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_200_sbert \
--device cuda:0 \
--percent_filename_suffix 200 \
--prompt sbert
```

```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_50_sbert \
--device cuda:0 \
--percent_filename_suffix 50 \
--prompt sbert
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_100_sbert \
--device cuda:0 \
--percent_filename_suffix 100 \
--prompt sbert
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_150_sbert \
--device cuda:0 \
--percent_filename_suffix 150 \
--prompt sbert
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_200_sbert \
--device cuda:0 \
--percent_filename_suffix 200 \
--prompt sbert
```

```
python3 transformers_trainer.py \
--model_folder models/conll/train_all \
--mode test \
--test_file dataset/ontonotes_conll/test.txt
```







```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_continual_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--checkpoint model_files/models/conll/train_all \
--model_folder models/ontonotes_conll/transfer_train_50_conll \
--device cuda:0 \
--percent_filename_suffix 50
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_continual_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--checkpoint model_files/models/conll/train_all \
--model_folder models/ontonotes_conll/transfer_train_100_max_conll \
--device cuda:0 \
--percent_filename_suffix 100
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_continual_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--checkpoint model_files/models/conll/train_all \
--model_folder models/ontonotes_conll/transfer_train_150_max_conll \
--device cuda:0 \
--percent_filename_suffix 150
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_continual_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--checkpoint model_files/models/conll/train_all \
--model_folder models/ontonotes_conll/transfer_train_200_max_conll \
--device cuda:0 \
--percent_filename_suffix 200
```


```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_continual_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--checkpoint model_files/models/conll/train_all \
--model_folder models/bc5cdr/transfer_train_50_max \
--device cuda:0 \
--percent_filename_suffix 50
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_continual_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--checkpoint model_files/models/conll/train_all \
--model_folder models/bc5cdr/transfer_train_100_max \
--device cuda:0 \
--percent_filename_suffix 100
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_continual_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--checkpoint model_files/models/conll/train_all \
--model_folder models/bc5cdr/transfer_train_150_max \
--device cuda:0 \
--percent_filename_suffix 150
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_continual_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--checkpoint model_files/models/conll/train_all \
--model_folder models/bc5cdr/transfer_train_200_max \
--device cuda:0 \
--percent_filename_suffix 200
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_continual_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--checkpoint model_files/models/conll/train_all \
--model_folder models/bc5cdr/transfer_train_200_max_new \
--device cuda:0 \
--percent_filename_suffix 200 \
--prompt max
```
### Slurm Commands
```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_all \
--device cuda:0 \
--percent_filename_suffix all
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_bertscore \
--device cuda:0 \
--percent_filename_suffix 50 \
--prompt bertscore
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_100_sbert \
--device cuda:0 \
--percent_filename_suffix 100 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_150_sbert \
--device cuda:0 \
--percent_filename_suffix 150 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_200_sbert \
--device cuda:0 \
--percent_filename_suffix 200 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_50_sbert \
--device cuda:0 \
--percent_filename_suffix 50 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_100_sbert \
--device cuda:0 \
--percent_filename_suffix 100 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_150_sbert \
--device cuda:0 \
--percent_filename_suffix 150 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_200_sbert \
--device cuda:0 \
--percent_filename_suffix 200 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_50_sbert \
--device cuda:0 \
--percent_filename_suffix 50 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_100_sbert \
--device cuda:0 \
--percent_filename_suffix 100 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_150_sbert \
--device cuda:0 \
--percent_filename_suffix 150 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_200_sbert \
--device cuda:0 \
--percent_filename_suffix 200 \
--prompt sbert
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_trainer.py \
--model_folder models/conll/train_all \
--mode test \
--test_file dataset/ontonotes_conll/test.txt
```







```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_continual_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--checkpoint model_files/models/conll/train_all \
--model_folder models/ontonotes_conll/transfer_train_50_conll \
--device cuda:0 \
--percent_filename_suffix 50
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_continual_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--checkpoint model_files/models/conll/train_all \
--model_folder models/ontonotes_conll/transfer_train_100_max_conll \
--device cuda:0 \
--percent_filename_suffix 100
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_continual_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--checkpoint model_files/models/conll/train_all \
--model_folder models/ontonotes_conll/transfer_train_150_max_conll \
--device cuda:0 \
--percent_filename_suffix 150
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_continual_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--checkpoint model_files/models/conll/train_all \
--model_folder models/ontonotes_conll/transfer_train_200_max_conll \
--device cuda:0 \
--percent_filename_suffix 200
```


```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_continual_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--checkpoint model_files/models/conll/train_all \
--model_folder models/bc5cdr/transfer_train_50_max \
--device cuda:0 \
--percent_filename_suffix 50
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_continual_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--checkpoint model_files/models/conll/train_all \
--model_folder models/bc5cdr/transfer_train_100_max \
--device cuda:0 \
--percent_filename_suffix 100
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_continual_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--checkpoint model_files/models/conll/train_all \
--model_folder models/bc5cdr/transfer_train_150_max \
--device cuda:0 \
--percent_filename_suffix 150
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_continual_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--checkpoint model_files/models/conll/train_all \
--model_folder models/bc5cdr/transfer_train_200_max \
--device cuda:0 \
--percent_filename_suffix 200
```

```
srun --gres=gpu:1080:1 --nodelist ink-lucy \
python3 transformers_continual_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--checkpoint model_files/models/conll/train_all \
--model_folder models/bc5cdr/transfer_train_200_max_new \
--device cuda:0 \
--percent_filename_suffix 200 \
--prompt max
```
