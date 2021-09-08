# 
# @author: Allan
#

import random
from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizer
import collections
import numpy as np
from src.data.data_utils import convert_iobes, build_label_idx, check_all_labels_in_dict

from src.data import Instance

Feature = collections.namedtuple('Feature', 'input_ids attention_mask token_type_ids orig_to_tok_index word_seq_len label_ids')
Feature.__new__.__defaults__ = (None,) * 6


def convert_instances_to_feature_tensors(instances: List[Instance],
                                         tokenizer: PreTrainedTokenizer,
                                         label2idx: Dict[str, int],
                                         prompt: str = None,
                                         prompt_candidates_from_outside: List[str] = None):
    features = []
    # max_candidate_length = -1
    entity_dict = None
    if prompt_candidates_from_outside is None and prompt is not None:
        ##### get popular entity
        entity_dict = {}
        for inst in instances:
            for entity, label in inst.entities:
                if label not in entity_dict:
                    entity_dict[label] = {}
                if entity not in entity_dict[label]:
                    entity_dict[label][entity] = [inst]
                else:
                    entity_dict[label][entity].append(inst)

        max_entities = {}
        for label in entity_dict:
            for x in sorted(entity_dict[label].items(), key=lambda kv: len(kv[1]), reverse=True)[0:1]:
                max_entities[label] = [x[0], random.choice(tuple(x[1])).words]

        ######
    elif prompt_candidates_from_outside is not None and prompt is not None:
        entity_dict = prompt_candidates_from_outside
        max_entities = {}
        for label in entity_dict:
            for x in sorted(entity_dict[label].items(), key=lambda kv: len(kv[1]), reverse=True)[0:1]:
                max_entities[label] = [x[0], random.choice(tuple(x[1])).words]

    for idx, inst in enumerate(instances):
        words = inst.ori_words
        orig_to_tok_index = []
        tokens = []
        for i, word in enumerate(words):
            """
            Note: by default, we use the first wordpiece token to represent the word
            If you want to do something else (e.g., use last wordpiece to represent), modify them here.
            """
            orig_to_tok_index.append(len(tokens))
            ## tokenize the word into word_piece / BPE
            ## NOTE: adding a leading space is important for BART/GPT/Roberta tokenization.
            ## Related GitHub issues:
            ##      https://github.com/huggingface/transformers/issues/1196
            ##      https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py#L38-L56
            ##      https://github.com/ThilinaRajapakse/simpletransformers/issues/458
            word_tokens = tokenizer.tokenize(" " + word)
            for sub_token in word_tokens:
                tokens.append(sub_token)
        labels = inst.labels
        label_ids = [label2idx[label] for label in labels] if labels else [-100] * len(words)

        if prompt is None:
            input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
        elif prompt == "max":
            prompt_tokens = []

            for entity_label in max_entities:
                # for i, word in enumerate(max_entities[entity_label][1]):
                #     word_tokens = tokenizer.tokenize(" " + word)
                #     for sub_token in word_tokens:
                #         prompt_tokens.append(sub_token)
                # prompt_tokens.append(tokenizer.sep_token)
                entity_tokens = tokenizer.tokenize(" " + max_entities[entity_label][0])
                for sub_token in entity_tokens:
                    prompt_tokens.append(sub_token)
                prompt_tokens.append("is")
                prompt_tokens.append(entity_label)
                # if entity_label == "PER":
                #     prompt_tokens.append("(")
                #     prompt_tokens.append("person")
                #     prompt_tokens.append(")")
                # elif entity_label == "ORG":
                #     prompt_tokens.append("(")
                #     prompt_tokens.append("organization")
                #     prompt_tokens.append(")")
                # elif entity_label == "LOC":
                #     prompt_tokens.append("(")
                #     prompt_tokens.append("location")
                #     prompt_tokens.append(")")
                # elif entity_label == "MISC":
                #     prompt_tokens.append("(")
                #     prompt_tokens.append("miscellaneous")
                #     prompt_tokens.append(")")
                # prompt_tokens.append(tokenizer.sep_token)

            # del prompt_tokens[-1]
            print(prompt_tokens)
            input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token] + prompt_tokens + [tokenizer.sep_token])

        elif prompt == "random":
            prompt_tokens = []
            for entity_label in entity_dict:

                entity = random.choice(tuple(entity_dict[entity_label]))
                entity_tokens = tokenizer.tokenize(" " + entity)
                for sub_token in entity_tokens:
                    prompt_tokens.append(sub_token)
                prompt_tokens.append("is")
                prompt_tokens.append(entity_label)

            print(prompt_tokens)

            input_ids = tokenizer.convert_tokens_to_ids(
                [tokenizer.cls_token] + tokens + [tokenizer.sep_token] + prompt_tokens + [tokenizer.sep_token])

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)


        features.append(Feature(input_ids=input_ids,
                                attention_mask=input_mask,
                                orig_to_tok_index=orig_to_tok_index,
                                token_type_ids=segment_ids,
                                word_seq_len=len(orig_to_tok_index),
                                label_ids=label_ids))

    if prompt_candidates_from_outside is None and prompt is not None:
        return features, entity_dict
    elif prompt_candidates_from_outside is not None and prompt is not None:
        return features
    else:
        return features


class TransformersNERDataset(Dataset):

    def __init__(self, file: str,
                 tokenizer: PreTrainedTokenizer,
                 is_train: bool,
                 sents: List[List[str]] = None,
                 label2idx: Dict[str, int] = None,
                 number: int = -1,
                 percentage: int = 100,
                 prompt:str = None,
                 prompt_candidates_from_outside: List[str] = None):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
        ## read all the instances. sentences and labels
        self.percentage = percentage
        insts = self.read_txt(file=file, number=number) if sents is None else self.read_from_sentences(sents)
        self.insts = insts
        if is_train:
            print(f"[Data Info] Using the training set to build label index")
            assert label2idx is None
            ## build label to index mapping. e.g., B-PER -> 0, I-PER -> 1
            idx2labels, label2idx = build_label_idx(insts)
            self.idx2labels = idx2labels
            self.label2idx = label2idx
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx
            self.label2idx = label2idx
            # check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)

        if is_train and prompt is not None:
            self.insts_ids, self.prompt_candidates = convert_instances_to_feature_tensors(insts, tokenizer, label2idx, prompt=prompt)
        else:
            self.insts_ids = convert_instances_to_feature_tensors(insts, tokenizer, label2idx, prompt=prompt, prompt_candidates_from_outside=prompt_candidates_from_outside)
        self.tokenizer = tokenizer


    def read_from_sentences(self, sents: List[List[str]]):
        """
        sents = [['word_a', 'word_b'], ['word_aaa', 'word_bccc', 'word_ccc']]
        """
        insts = []
        for sent in sents:
            insts.append(Instance(words=sent, ori_words=sent))
        return insts


    def read_txt(self, file: str, number: int = -1) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}, labels will be converted to IOBES encoding")
        print(f"[Data Info] Modify src/data/transformers_dataset.read_txt function if you have other requirements")
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            ori_words = []
            labels = []
            entities = []
            entity = []
            entity_label = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    labels = convert_iobes(labels)
                    if len(entity) != 0:
                        entities.append([" ".join(entity),entity_label[0]])
                    if len(set(labels)) > 1:
                        insts.append(Instance(words=words, ori_words=ori_words, labels=labels, entities=entities))
                    words = []
                    ori_words = []
                    labels = []
                    entities = []
                    entity = []
                    entity_label = []
                    if len(insts) == number:
                        break
                    continue
                ls = line.split()
                word, label = ls[0],ls[-1]
                ori_words.append(word)
                words.append(word)
                labels.append(label)

                if label.startswith("B"):
                    entity.append(word)
                    entity_label.append(label.split('-')[1])
                elif label.startswith("I"):
                    entity.append(word)
                else:
                    if len(entity) != 0:
                        entities.append([" ".join(entity), entity_label[0]])
                        entity = []
                        entity_label = []

        numbers = int(len(insts) * self.percentage / 100)
        percentage_insts = insts[:numbers]

        print("number of sentences: {}".format(len(percentage_insts)))
        return percentage_insts

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_fn(self, batch:List[Feature]):
        word_seq_len = [len(feature.orig_to_tok_index) for feature in batch]
        max_seq_len = max(word_seq_len)
        max_wordpiece_length = max([len(feature.input_ids) for feature in batch])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            input_ids = feature.input_ids + [self.tokenizer.pad_token_id] * padding_length
            mask = feature.attention_mask + [0] * padding_length
            type_ids = feature.token_type_ids + [self.tokenizer.pad_token_type_id] * padding_length
            padding_word_len = max_seq_len - len(feature.orig_to_tok_index)
            orig_to_tok_index = feature.orig_to_tok_index + [0] * padding_word_len
            label_ids = feature.label_ids + [0] * padding_word_len

            batch[i] = Feature(input_ids=np.asarray(input_ids),
                               attention_mask=np.asarray(mask), token_type_ids=np.asarray(type_ids),
                               orig_to_tok_index=np.asarray(orig_to_tok_index),
                               word_seq_len =feature.word_seq_len,
                               label_ids=np.asarray(label_ids))
        results = Feature(*(default_collate(samples) for samples in zip(*batch)))
        return results


## testing code to test the dataset
# from transformers import *
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# dataset = TransformersNERDataset(file= "data/conll2003_sample/train.txt",tokenizer=tokenizer, is_train=True)
# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2, collate_fn=dataset.collate_fn)
# print(len(train_dataloader))
# for batch in train_dataloader:
#     # print(batch.input_ids.size())
#     print(batch.input_ids)
#     pass
