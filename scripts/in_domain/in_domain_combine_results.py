import argparse
import numpy as np
from collections import defaultdict


def get_parser():
    parser = argparse.ArgumentParser(
        description="retrieve with bart")

    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset")
    parser.add_argument('--prompt', type=str, required=True,
                        help="Prompt")
    parser.add_argument('--template', type=str, required=True,
                        help="Template")
    parser.add_argument('--shots', type=int, required=True,
                        help="shots")
    parser.add_argument('--output_file', type=str, required=True,
                        help="output to a txt")

    return parser


def main():
    args = get_parser().parse_args()

    def get_path(sample_seed, train_seed):
        """
            Path to result txt file
        """
        return f"results_metrics/{args.dataset}/bert_base_cased_crf/train_{args.prompt}_{args.template}_{args.shots}_{sample_seed}_{train_seed}.txt"

    def get_score(sample_seed, train_seed):
        with open(get_path(sample_seed, train_seed)) as fd:
            score = float(fd.readlines()[-1].split(":")[-1].strip())
        return score

    avgs = []
    all_scores = {}
    for sample_seed in [42, 1337, 2021, 5555, 9999]:
        sample_scores = []
        for train_seed in [42, 1337, 2021]:
            s = get_score(sample_seed, train_seed)
            sample_scores.append(s)
            all_scores[f"TrainSeed_{train_seed}_SampleSeed_{sample_seed}"] = s
        avgs.append(np.mean(sample_scores))

    with open(args.output_file, "w") as fd:
        for k,v in all_scores.items():
            fd.write(f"{k} has F1 {v}\n")
        fd.write(f"Average: {np.mean(list(all_scores.values()))}\n")
        fd.write(f"Std: {np.std(avgs)}\n")
    
    print(f"Saved combined results to {args.output_file}")


if __name__ == "__main__":
    main()
