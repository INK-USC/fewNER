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
parser.add_argument('--constraint', action='store_true', required=False, default=False, help='constraint instances')

args = parser.parse_known_args()[0]

suffices = ([args.suffix + "_1337", args.suffix + "_2021", args.suffix + "_5555", args.suffix + "_42", args.suffix + "_9999"] if not args.no_subsamples else [args.suffix])
seeds = ['42', '1337', '2021']
print(f"\nUsing suffices: {suffices}")
sys.stdout.flush()

try:
    os.makedirs("logs/" + args.dataset + "/")
except:
    pass

constraint = "constraint_" if args.constraint else ""

def gen_command_logfilename_modelfolder(args, seed, suffix):
    """
        Returns (command, log file, model folder)
    """
    log_file = "logs/" + args.dataset + "/" + constraint + args.train_file.split('.')[0] + "_" + (args.prompt if args.prompt else "None") + "_" + (args.template if args.prompt else "None") + "_" + suffix + "_" + seed + ".txt"
    model_folder = "models/" + args.dataset + "/" + constraint + args.train_file.split('.')[0] + "_" + (args.prompt if args.prompt else "None") +"_" + (args.template if args.prompt else "None") +  "_"  + suffix + "_" + seed
    predict_cmd = None
    predict_cmd = \
        ("" if args.constraint else " CONST_INST=0") + \
        " python3 " + args.train_file + \
        " --dataset " + args.dataset + \
        " --data_dir " + args.data_dir + \
        " --model_folder " + model_folder + \
        " --device cuda:0" + \
        " --percent_filename_suffix " + suffix + \
        " --num_epochs 50" + \
        " --batch_size 4" + \
        ((" --prompt " + args.prompt + " --template " + args.template) if args.prompt else "") + \
        " --seed " + seed + " > " + log_file

    return predict_cmd,log_file,model_folder


log_files = []
model_folders = []
for suffix in suffices:
    sub_runs = []
    for seed in seeds:
        predict_cmd,log_file,model_folder = gen_command_logfilename_modelfolder(args, seed, suffix)
        sub_runs.append(log_file)
        model_folders.append(model_folder)
        print("Executing command: " + predict_cmd + "\n")
        sys.stdout.flush() # Otherwise the command won't show for some reason
        # if not os.path.exists(model_folder):
        os.system(predict_cmd)
    log_files.append(sub_runs)

for model_folder in model_folders:
    rm_cmd = "rm -rf model_files/" + model_folder
    os.system(rm_cmd)

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
f1_scores = []
for subrun in log_files:
    subrun_f1s = []
    for file in subrun:
        with open(file, 'r') as reader:
            for line in reader:
                pass
            last_line = line
            subrun_f1s.append(float(ansi_escape.sub('', last_line.split()[-1])))
    f1_scores.append(subrun_f1s)

print(f"F1 score array {f1_scores}")

arr = numpy.array(f1_scores)
mean = arr.flatten().mean()
# Calculate the std of the average instead of everything
std = arr.mean(axis=1).std()
print("average: ", mean)
print("std: ", std)

# Write result summary to a txt file. 
with open(args.dataset + "_" + constraint +  args.train_file.split('.')[0] + "_" + (args.prompt if args.prompt else "None") + "_" + (args.template if args.prompt else "None") + "_" + args.suffix + ".txt", 'w') as file:
    file.write("average: " + str(mean))
    file.write("std: " + str(std))
