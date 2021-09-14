import random
import argparse
import numpy
random.seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('--inp_data', type=str, default='dataset/fewnerd/supervised/dev.txt', help="text file - dataset")
parser.add_argument('--out_data', type=str, default='dataset/fewnerd/supervised/converted/dev.txt', help="text file - dataset")


args = parser.parse_known_args()[0]
print(args)

inp_data=args.inp_data
out_data=args.out_data

with open(inp_data,'r',encoding='utf8') as f:
    lines=f.readlines()

arr = []
sent = []
inside = False
for i in range(len(lines)):
    if lines[i] =='\n':
        arr.append(sent)
        sent=[]
    else:
        
        word = lines[i].strip().split()[0]
        label = lines[i].strip().split()[-1]
        new_label =''
        if label=='O':
            new_label='O'
            inside = False
        elif inside == False or i==0:
            # temp = label
            new_label = 'B-'+label
            inside = True
        elif inside == True and label==lines[i-1].strip().split()[-1]:
            # print(temp)
            new_label = 'I-'+label
        sent.append([word,new_label])

  
with open(out_data,'w',encoding='utf8') as f:
    for sent in arr:
        for line in sent:
            f.write(line[0]+' '+line[1]+'\n')
            
        f.write('\n')

# print(arr)


    # inside = False
    # label = line.strip().split()[-1]

    # words = [t["text"] for t in sent["tokens"]]
    # labels = ['O'] * len(words)
    # for e in sent["basic_events"]:
    #      event_type = e["event_type"]
    #      for span in e["anchors"]["spans"]:
    #          start = span["grounded_span"]["full_span"]["start_token"]
    #          end = span["grounded_span"]["full_span"]["end_token"] + 1
    #          labels[start] = f'B-{event_type}'
    #          for idx in range(start + 1, end):
    #              labels[idx] = f'I-{event_type}'
    # return words, labels