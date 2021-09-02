python train.py \
--data-dir 'dataset/CONLL' \
--model-type 'bert' \
--model-name 'bert-large-cased' \
--output-dir 'eval/conll2003' \
--gpu '0,1,2,3' \
--labels 'dataset/CONLL/labels.txt' \
--max-seq-length 128 \
--overwrite-output-dir \
--do-train \
--do-eval \
--do-predict \
--evaluate-during-training \
--batch-size 8 \
--learning-rate 5e-5 \
--gradient-accumulation-steps 4 \
--num-train-epochs 20 \
--save-steps 750 \
--seed 1 \
--train-examples -1 \
--eval-batch-size 128 \
--pad-subtoken-with-real-label \
--eval-pad-subtoken-with-first-subtoken-only \
--label-sep-cls \
--trigger