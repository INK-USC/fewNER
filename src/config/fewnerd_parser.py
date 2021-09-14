import argparse


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
        
            new_label = 'B-'+label
            inside = True
        elif inside == True and label==lines[i-1].strip().split()[-1]:
        
            new_label = 'I-'+label
        sent.append([word,new_label])

  
with open(out_data,'w',encoding='utf8') as f:
    for sent in arr:
        for line in sent:
            f.write(line[0]+' '+line[1]+'\n')
            
        f.write('\n')

