CUDA_VISIBLE_DEVICES=1 python3 transformers_trainer.py --dataset conll --data_dir dataset/conll/human_triggers --model_folder models/conll/baseline --device cuda:0


CUDA_VISIBLE_DEVICES=4 python3 transformers_trainer_concat.py --dataset conll --data_dir dataset/conll/ner --model_folder models/conll/entityconcat20156 --device cuda:0 --percent_filename_suffix 201
CUDA_VISIBLE_DEVICES=4 python3 transformers_trainer.py --dataset bc5cdr --data_dir dataset/bc5cdr/ner --model_folder models/bc5cdr/na --device cuda:0 --percent_filename_suffix 251
