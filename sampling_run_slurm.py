import os
import re
import argparse
import numpy
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, required=True, help='Finetuning file')
parser.add_argument('--dataset', type=str, required=True, help='Dataset')
parser.add_argument('--data_dir', type=str, required=True, help='Data Directory')
parser.add_argument('--suffix', type=str, required=True, help='Size of training file')
parser.add_argument('--prompt', type=str, required=False, help='Selection strategy')
parser.add_argument('--template', type=str, required=False, help='Template')
parser.add_argument('--no_subsamples', type=bool, required=False, default=False, help='No seeded subsamples is this is specified (any value will evaluate to true)')

args = parser.parse_known_args()[0]

suffices = ([args.suffix + "_1337", args.suffix + "_2021", args.suffix + "_5555", args.suffix + "_42", args.suffix] if not args.no_subsamples else [args.suffix])
seeds = ['42', '1337', '2021']
print(f"\nUsing suffices: {suffices}")
sys.stdout.flush()

try:
    os.makedirs("logs/" + args.dataset + "/")
except:
    pass


def gen_command_logfilename_modelfolder(args, seed, suffix):
    """
        Returns (command, log file, model folder)
    """
    log_file = "logs/" + args.dataset + "/" + args.train_file.split('.')[0] + "_" + (args.prompt if args.prompt else "None") + "_" + (args.template if args.prompt else "None") + "_" + suffix + "_" + seed + ".txt"
    model_folder = "models/" + args.dataset + "/" + args.train_file.split('.')[0] + "_" + (args.prompt if args.prompt else "None") +"_" + (args.template if args.prompt else "None") +  "_"  + suffix + "_" + seed
    predict_cmd = None
    predict_cmd = \
        " python3 " + args.train_file + \
        " --dataset " + args.dataset + \
        " --data_dir " + args.data_dir + \
        " --model_folder " + model_folder + \
        " --device cuda:0" + \
        " --percent_filename_suffix " + suffix + \
        " --num_epochs 50" + \
        ((" --prompt " + args.prompt + " --template " + args.template) if args.prompt else "") + \
        " --seed " + seed + " > " + log_file

    return predict_cmd,log_file,model_folder


log_files = []
model_folders = []
for suffix in suffices:
    for seed in seeds:
        predict_cmd,log_file,model_folder = gen_command_logfilename_modelfolder(args, seed, suffix)
        log_files.append(log_file)
        model_folders.append(model_folder)
        print("Executing command: " + predict_cmd + "\n")
        sys.stdout.flush() # Otherwise the command won't show for some reason
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

# Write result summary to a txt file. 
with open(args.dataset + "_" + args.train_file.split('.')[0] + "_" + (args.prompt if args.prompt else "None") + "_" + (args.template if args.prompt else "None") + "_" + args.suffix + ".txt", 'w') as file:
    file.write("average: " + str(mean))
    file.write("std: " + str(std))
