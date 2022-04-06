dataset=ontonotes_conll
shots=25
prompt=sbert
template=context_all
check_point=checkpoint/conll_all
# 15 runs in total
for sample_seed in 42 1337 2021 5555 9999;
do
    for train_seed in 42 2021 1337;
    do 
        sbatch scripts/domain_adaptation/domain_adaptation_one.sh ${dataset} ${shots} ${prompt} ${template} ${train_seed} ${sample_seed} ${check_point}
    done
done
