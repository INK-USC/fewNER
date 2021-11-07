import argparse
import random
import numpy as np
from src.config import Config, evaluate_batch_insts
import time
from src.model import TransformersCRF
import torch
from typing import List
from termcolor import colored
from src.config.utils import write_results
from src.config.transformers_util import get_huggingface_optimizer_and_scheduler
from src.config import context_models, get_metric
import pickle
import tarfile
from tqdm import tqdm
from collections import Counter
from src.data import TransformersNERDataset, SearchSpaceManager
from torch.utils.data import DataLoader
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hardcode a file to see whether the search space manager is finding the right search space.
tokenizer = context_models["bert-base-cased"]["tokenizer"].from_pretrained("bert-base-cased")
dataset = TransformersNERDataset("dataset/conll/train_200.txt", tokenizer, number=-1, is_train=True, percentage=100)
manager = SearchSpaceManager(dataset.insts)
x = dataset.insts[1]
print(len(dataset.insts))
print(x)

sspace = manager.single_label_search_space("PER")
cspace = manager.superset_labels_search_space(x)

# for ins in sspace:
#     print(set(lb for e,lb in ins.entities))

print("----------------------")

for ins in cspace:
    print(set(lb for e,lb in ins.entities))

print(len(sspace), len(cspace))


