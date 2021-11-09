from os import listdir, remove
from os.path import isfile, join
import shutil
mdir = "./model_files/models/conll/"
mdir = "./logs/conll/"
donwant = {'P6'}
dfs = [f for f in listdir(mdir) if (f.split('.')[0].split('_')[-1] in donwant)]

print(*(f for f in dfs), sep="\n")
x = input("Are these the files you want to delete? (y/n)")
if x is not 'y':
    exit(0)

for f in dfs:
    if isfile(mdir + f):
        remove(mdir + f)
    else:
        shutil.rmtree(mdir + f)

print("Deleted!")
