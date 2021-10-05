import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, required=True, help='Finetuning file')
parser.add_argument('--dataset', type=str, required=True, help='Dataset')
parser.add_argument('--data_dir', type=str, required=True, help='Data Directory')
parser.add_argument('--suffix', type=str, required=True, help='Data Directory')
parser.add_argument('--gpu', type=str, default="0", help='GPU ids separated by "," to use for computation')

args = parser.parse_known_args()[0]

suffices = [args.suffix, args.suffix + "_1337", args.suffix + "_2021"]
seeds = ['42', '1337', '2021']

for suffix in suffices:
    for seed in seeds:
        log_file = "logs/" + args.dataset + "/" + args.train_file.split('.')[0] + "_" + suffix + "_" + seed + ".txt"
        model_folder = "models/" + args.dataset + "/" + args.train_file.split('.')[0] + "_" + suffix + "_" + seed
        predict_cmd = \
            "CUDA_VISIBLE_DEVICES=" + str(args.gpu) + \
            " python3 " + args.train_file + \
            " --dataset " + args.dataset + \
            " --data_dir " + args.data_dir + \
            " --model_folder " + model_folder + \
            " --device cuda:0" + \
            " --percent_filename_suffix " + suffix + \
            " --num_epochs 50" + \
            " --seed " + seed + " > " + log_file

        print(predict_cmd, "\n")
        os.system(predict_cmd)
