```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50 \
--device cuda:0 \
--percent_filename_suffix 50
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_100 \
--device cuda:0 \
--percent_filename_suffix 100
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_150 \
--device cuda:0 \
--percent_filename_suffix 150
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_200 \
--device cuda:0 \
--percent_filename_suffix 200
```

```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_50 \
--device cuda:0 \
--percent_filename_suffix 50
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_100 \
--device cuda:0 \
--percent_filename_suffix 100
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_150 \
--device cuda:0 \
--percent_filename_suffix 150
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_200 \
--device cuda:0 \
--percent_filename_suffix 200
```

```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_50 \
--device cuda:0 \
--percent_filename_suffix 50
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_100 \
--device cuda:0 \
--percent_filename_suffix 100
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_150 \
--device cuda:0 \
--percent_filename_suffix 150
```

```
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_200 \
--device cuda:0 \
--percent_filename_suffix 200
```
