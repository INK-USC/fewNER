CUDA_VISIBLE_DEVICES=0 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_default_42 \
--device cuda:0 \
--percent_filename_suffix 50 \
--num_epochs 50 \
--seed 42

CUDA_VISIBLE_DEVICES=0 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_1337_42 \
--device cuda:0 \
--percent_filename_suffix 50_1337 \
--num_epochs 50 \
--seed 42

CUDA_VISIBLE_DEVICES=0 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_2021_42 \
--device cuda:0 \
--percent_filename_suffix 50_2021 \
--num_epochs 50 \
--seed 42

CUDA_VISIBLE_DEVICES=0 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_default_1337 \
--device cuda:0 \
--percent_filename_suffix 50 \
--num_epochs 50 \
--seed 1337

CUDA_VISIBLE_DEVICES=0 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_1337_1337 \
--device cuda:0 \
--percent_filename_suffix 50_1337 \
--num_epochs 50 \
--seed 1337

CUDA_VISIBLE_DEVICES=0 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_2021_1337 \
--device cuda:0 \
--percent_filename_suffix 50_2021 \
--num_epochs 50 \
--seed 1337

CUDA_VISIBLE_DEVICES=0 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_default_2021 \
--device cuda:0 \
--percent_filename_suffix 50 \
--num_epochs 50 \
--seed 2021

CUDA_VISIBLE_DEVICES=0 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_1337_2021 \
--device cuda:0 \
--percent_filename_suffix 50_1337 \
--num_epochs 50 \
--seed 2021

CUDA_VISIBLE_DEVICES=0 \
python3 transformers_trainer.py \
--dataset conll \
--data_dir dataset/conll \
--model_folder models/conll/train_50_2021_2021 \
--device cuda:0 \
--percent_filename_suffix 50_2021 \
--num_epochs 50 \
--seed 2021
