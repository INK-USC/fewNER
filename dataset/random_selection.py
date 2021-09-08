import random
import argparse
import numpy
random.seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='ontonotes_conll/train.txt', help="text file - dataset")
parser.add_argument('--number', type=int, default=50, help="number of sentences in dataset")

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

B_PREF="B-"
I_PREF = "I-"
S_PREF = "S-"
E_PREF = "E-"
O = "O"

def convert_iobes(labels):
	for pos in range(len(labels)):
		curr_entity = labels[pos]
		if pos == len(labels) - 1:
			if curr_entity.startswith(B_PREF):
				labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				labels[pos] = curr_entity.replace(I_PREF, E_PREF)
		else:
			next_entity = labels[pos + 1]
			if curr_entity.startswith(B_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(B_PREF, S_PREF)
			elif curr_entity.startswith(I_PREF):
				if next_entity.startswith(O) or next_entity.startswith(B_PREF):
					labels[pos] = curr_entity.replace(I_PREF, E_PREF)
	return labels

def labels_from_sents(sents):
    labels=set()
    for sent in sents:
        sent_label = []
        for i in sent:
            label=i.split(' ')[-1]
            sent_label.append(label)
        sent_label = convert_iobes(sent_label)
        for s_label in sent_label:
            labels.add(s_label)
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
    sents = data_to_sents(data)
    labels = set()

    sent_index = 0
    sliced_sents = []
    while len(list(labels)) != len(label_space):
        if len(sliced_sents) == number:
            del sliced_sents[-1]
        sent_labels = []
        for s in sents[sent_index]:
            sent_labels.append(s.split(' ')[-1])
        sent_labels = convert_iobes(sent_labels)
        if len(list(set(sent_labels))) > 1:
            labels = labels.union(set(sent_labels))
            sliced_sents.append(sents[sent_index])
        sent_index += 1

    return sliced_sents


all_sents=data_to_sents(data)
label_space=labels_from_sents(all_sents)
reduced_sents=dataset_slice(number,lines,label_space)

refined_file = open(f'ontonotes_conll/train_50.txt', 'w')

def write_original(refined, writefile):
    for instance in refined:
        writefile.writelines(instance)
        writefile.write('\n')

write_original(reduced_sents, refined_file)

