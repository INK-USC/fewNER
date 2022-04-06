dataset=conll
shots=25
prompt=max
template=context
# 15 runs in total
for sample_seed in 42 1337 2021 5555 9999;
do
    for train_seed in 42 2021 1337;
    do 
        sbatch scripts/in_domain/in_domain_one.sh ${dataset} ${shots} ${prompt} ${template} ${train_seed} ${sample_seed}
    done
done
