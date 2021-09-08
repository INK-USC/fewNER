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
--model_folder models/conll/train_50_max2 \
--device cuda:0 \
--percent_filename_suffix 50
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_100_random \
--device cuda:0 \
--percent_filename_suffix 100
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_150_random \
--device cuda:0 \
--percent_filename_suffix 150
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_200_max2 \
--device cuda:0 \
--percent_filename_suffix 200
```

```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_50_random \
--device cuda:0 \
--percent_filename_suffix 50
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_100_random \
--device cuda:0 \
--percent_filename_suffix 100
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_150_random \
--device cuda:0 \
--percent_filename_suffix 150
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset bc5cdr \
--data_dir dataset/bc5cdr \
--model_folder models/bc5cdr/train_200_random \
--device cuda:0 \
--percent_filename_suffix 200
```

```
CUDA_VISIBLE_DEVICES=1 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_50_conll \
--device cuda:0 \
--percent_filename_suffix 50
```

```
CUDA_VISIBLE_DEVICES=2 \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_100_conll \
--device cuda:0 \
--percent_filename_suffix 100
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_150_conll\
--device cuda:0 \
--percent_filename_suffix 150
```

```
CUDA_VISIBLE_DEVICES=3 \
python3 transformers_trainer.py \
--dataset ontonotes_conll \
--data_dir dataset/ontonotes_conll \
--model_folder models/ontonotes_conll/train_200_conll \
--device cuda:0 \
--percent_filename_suffix 200
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
