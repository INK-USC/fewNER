import os
import re
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, required=True, help='Finetuning file')
parser.add_argument('--dataset', type=str, required=True, help='Dataset')
parser.add_argument('--data_dir', type=str, required=True, help='Data Directory')
parser.add_argument('--suffix', type=str, required=True, help='Data Directory')
parser.add_argument('--prompt', type=str, required=True, help='Data Directory')
parser.add_argument('--template', type=str, required=True, help='Data Directory')
parser.add_argument('--n_shot', type=str, default='1', required=False, help='N-shot value')

args = parser.parse_known_args()[0]

suffices = [args.suffix, args.suffix + "_1337", args.suffix + "_2021"]
seeds = ['42', '1337', '2021']

try:
    os.makedirs("logs/" + args.dataset + "/")
except:
    pass

log_files = []
model_folders = []
for suffix in suffices:
    for seed in seeds:
        log_file = "logs/" + args.dataset + "/" + args.train_file.split('.')[0] + "_" + args.prompt + "_" + args.template + "_" + suffix + "_" + seed + "_"+args.n_shot+".txt"
        model_folder = "models/" + args.dataset + "/" + args.train_file.split('.')[0] + "_" + args.prompt + "_" + args.template + "_"  + suffix + "_" + seed +"_"+args.n_shot
        predict_cmd = \
            " python3 " + args.train_file + \
            " --dataset " + args.dataset + \
            " --data_dir " + args.data_dir + \
            " --model_folder " + model_folder + \
            " --device cuda:0" + \
            " --percent_filename_suffix " + suffix + \
            " --num_epochs 50" + \
            " --prompt " + args.prompt + \
            " --template " + args.template + \
            " --n_shot " + args.n_shot + \
            " --seed " + seed + " > " + log_file
            
        log_files.append(log_file)
        model_folders.append(model_folder)
        print(predict_cmd, "\n")
        os.system(predict_cmd)

for model_folder in model_folders:
    rm_cmd = "rm -rf model_files/" + model_folder
    os.system(rm_cmd)

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
f1_scores = []
for file in log_files:
    with open(file, 'r') as reader:
        for line in reader:
            pass
        last_line = line
        f1_scores.append(float(ansi_escape.sub('', last_line.split()[-1])))

arr = numpy.array(f1_scores)
mean = numpy.mean(arr, axis=0)
std = numpy.std(arr, axis=0)
print("average: ", numpy.mean(arr, axis=0))
print("std: ", numpy.std(arr, axis=0))

with open(args.dataset + "_" + args.train_file.split('.')[0] + "_" + args.prompt + "_" + args.template + "_" + args.suffix +"_" +args.n_shot+".txt", 'w') as file:
    file.write("average: " + str(mean))
    file.write("std: " + str(std))
