import random
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='conll/train_all.txt', help="text file - dataset")
parser.add_argument('--target_data', type=str, default='conll/train_50_5555.txt', help="text file - dataset")
parser.add_argument('--seed', type=int, default=None, help="number of sentences in dataset")

args = parser.parse_known_args()[0]

data = args.data
seed = args.seed

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
            label = i.split('\t')[-1]
            sent_label.append(label)
        sent_label = convert_iobes(sent_label)
        for s_label in sent_label:
            labels.add(s_label)
    labels=list(labels)
    return labels

def dataset_slice(sents, label_space):
    sp = random.sample(sents, len(sents)//2)
    return sp

sents = []
temp = []
with open(data,'r') as f:
    lines=f.readlines()
    for i in lines:
        if i == '\n':
            sents.append(temp)
            temp=[]
        else:
            temp.append(i)
print(f"Read {len(sents)} sentences")
print(f"{sents[0]}")



label_space=labels_from_sents(sents)
reduced_sents=dataset_slice(sents, label_space)

refined_file = open(args.target_data, 'w')
 
def write_original(refined, writefile):
    for instance in refined:
        writefile.writelines(instance)
        writefile.write('\n')

write_original(reduced_sents, refined_file)

