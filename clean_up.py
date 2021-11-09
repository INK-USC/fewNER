from os import listdir, remove
from os.path import isfile, join
import shutil
mdir = "./model_files/models/conll/"
# mdir = "./logs/conll/"
bsx = "25"

ids = range(21,24)
donwant = set()
for i in ids:
    donwant.add("P" + str(i))
print(f"Don't want: {donwant}")

dfs = [f for f in listdir(mdir) if (f.split('.')[0].split('_')[-1] in donwant and f.split('.')[0].split('_')[-4] == bsx)]

print(*(f for f in dfs), sep="\n")
x = input(f"Are these the files you want to delete all these {len(dfs)} files? (y/n)")
if x is not 'y':
    exit(0)

for f in dfs:
    if isfile(mdir + f):
        remove(mdir + f)
    else:
        shutil.rmtree(mdir + f)

print("Deleted!")
