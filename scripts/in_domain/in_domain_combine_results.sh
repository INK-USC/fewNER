dataset=conll
shots=25
prompt=max
template=context

output_file="combine_results_${prompt}_${template}_${shots}.txt"

python3 ./scripts/in_domain/in_domain_combine_results.py \
    --dataset $dataset \
    --shots $shots \
    --template $template \
    --prompt $prompt \
    --output_file $output_file 
