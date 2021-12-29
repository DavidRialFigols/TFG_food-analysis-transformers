import os
from glob import glob

folder = "./food-101"
train_file = folder + "/meta/train.txt" 
test_file = folder + "/meta/test.txt"

# create folders
os.system(f"mkdir {folder}/train")
os.system(f"mkdir {folder}/test")
for cat in glob(f"{folder}/images/*"):
    os.system(f"mkdir {folder}/train/{cat[cat.rfind('/')+1:]}")
    os.system(f"mkdir {folder}/test/{cat[cat.rfind('/')+1:]}")

# move images
ftrain = open(train_file, 'r')
for line in ftrain.readlines():
    os.system(f"mv {folder}/images/{line[:-1]}.jpg {folder}/train/{line[:-1]}.jpg")
ftrain.close()
ftest = open(test_file, 'r')
for line in ftest.readlines():
    os.system(f"mv {folder}/images/{line[:-1]}.jpg {folder}/test/{line[:-1]}.jpg")
ftest.close()

# rm old folders
os.system(f"rm -r {folder}/images")
os.system(f"rm -r {folder}/meta")