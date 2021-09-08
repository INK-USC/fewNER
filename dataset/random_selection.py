import random
import argparse
import numpy
random.seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='bc5cdr/train.txt', help="text file - dataset")
parser.add_argument('--number', type=int, default=100, help="number of sentences in dataset")

args = parser.parse_known_args()[0]
print(args)

data=args.data
number=args.number

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


def dataset_slice(number,data,label_space):
    sents=data_to_sents(data)
    #random.shuffle(sents)
    labels=labels_from_sents(sents)
    counter=count_labels(sents)
    sents=sents[:number]
    if len(labels)==len(label_space):
        return sents
    else:
        temp=[i for i in labels + label_space if i not in labels or i not in label_space]
        for i in temp:
            for j in range(len(data)):
                if i in data[j]:
                    sents.append(data[j])
                break
    num_extra=len(sents)-number
    for label,count in counter.items():
        if num_extra<count:
            del_label=key
    for i in sents:
        if num_extra!=0:
            if del_label in i:
                sents.remove(i)
        else:
            break   
    return sents


all_sents=data_to_sents(data)
label_space=labels_from_sents(all_sents)
reduced_sents=dataset_slice(number,lines,label_space)

refined_file = open(f'bc5cdr/train_100.txt', 'w')

def write_original(refined, writefile):
    for instance in refined:
        writefile.writelines(instance)
        writefile.write('\n')

write_original(reduced_sents, refined_file)

