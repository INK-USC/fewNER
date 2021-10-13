import random
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='bc5cdr/train.txt', help="text file - dataset")
parser.add_argument('--target_data', type=str, default='conll/train_50_5555.txt', help="text file - dataset")
parser.add_argument('--number', type=int, default=50, help="number of sentences in dataset")
parser.add_argument('--seed', type=int, help="number of sentences in dataset")

args = parser.parse_known_args()[0]

data = args.data
number = args.number
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

def dataset_slice(number, sents, label_space):
    random.seed(seed)
    random.shuffle(sents)
    labels = set()

    sent_index = 0
    sliced_sents = []
    while (len(list(labels)) != len(label_space)) or (len(sliced_sents) < number):
        if len(sliced_sents) == number:
            delete_index = 0
            for i, sent in enumerate(sliced_sents):
                tmp_bool = False
                tmp_labels = []
                for s in sent:
                    tmp_labels.append(s.split(' ')[-1])
                tmp_labels = convert_iobes(tmp_labels)
                for tl in tmp_labels:
                    if tl.startswith("I-"):
                        tmp_bool = True
                if not tmp_bool:
                    delete_index = i
                    break

            del sliced_sents[delete_index]

        sent_labels = []
        for s in sents[sent_index]:
            sent_labels.append(s.split('\t')[-1])
        sent_labels = convert_iobes(sent_labels)
        if len(list(set(sent_labels))) > 1:
            labels = labels.union(set(sent_labels))
            sliced_sents.append(sents[sent_index])
        sent_index += 1

    print(label_space)
    print(labels)
    print(len(sliced_sents))

    return sliced_sents

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

label_space=labels_from_sents(sents)
reduced_sents=dataset_slice(number, sents, label_space)

refined_file = open(args.target_data, 'w')

def write_original(refined, writefile):
    for instance in refined:
        writefile.writelines(instance)
        writefile.write('\n')

write_original(reduced_sents, refined_file)

