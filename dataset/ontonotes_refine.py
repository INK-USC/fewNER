import random
import argparse
import numpy
random.seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='ontonotes/test.txt', help="text file - dataset")

args = parser.parse_known_args()[0]
print(args)

data=args.data

with open(data,'r') as f:
    lines=f.readlines()

def data_to_sents(data):
    sents=[]
    temp=[]
    for i in lines:
        if i=='\n':
            sents.append(temp)
            temp=[]
        else:
            temp.append(i)
    return sents

def labels_from_sents(sents):
    labels=set()
    sents=list(numpy.concatenate(sents).flat)
    counter={}
    for i in sents:
        label=i.split(' ')[-1]
        if label not in labels:
            labels.add(label)
    labels=list(labels)
    return labels

def count_labels(sents):
    counter={}
    sents=list(numpy.concatenate(sents).flat)
    for i in sents:
        label=i.split(' ')[-1]
        if label not in counter:
            counter[label]=1
        else:
            counter[label]+=1
    return counter


def clean_labels(labels):
    for i in range(len(labels)):
        labels[i]=labels[i].replace('\n','')
        if '-' in labels[i]:
            labels[i]=labels[i].split('-')[-1]
    if '' in labels:
        labels.remove('')
    return list(set(labels))

refined_file = open(f'ontonotes_conll/test_new.txt', 'w')

all_sents=data_to_sents(data)
for sent in all_sents:
    for i in sent:
        label = i.split(' ')[-1].replace('\n','')
        new_label = None
        if label != "O":
            if label == "B-PERSON":
                new_label = "B-PER"
            elif label == "I-PERSON":
                new_label = "I-PER"
            elif label == "B-GPE":
                new_label = "B-LOC"
            elif label == "I-GPE":
                new_label = "I-LOC"
            elif label == "B-ORG":
                new_label = "B-ORG"
            elif label == "I-ORG":
                new_label = "I-ORG"
            else:
                if label.startswith("B-"):
                    new_label = "B-MISC"
                elif label.startswith("I-"):
                    new_label = "I-MISC"
        else:
            new_label = "O"

        refined_file.writelines(i.split(' ')[0] + ' ' + new_label)
        refined_file.write('\n')
    refined_file.write('\n')
