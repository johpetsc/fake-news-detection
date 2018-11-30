import pandas as pd
import os

#header = ["0", "1", "2", "3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"]
dir1 = 'Fake.br-Corpus/full_texts/fake-meta-information'
dir2 = 'Fake.br-Corpus/full_texts/true-meta-information'
filelist1 = os.listdir(dir1) 
filelist2 = os.listdir(dir2)

data1 = pd.concat([pd.read_table(dir1+'/'+item,delimiter="\n",sep=" ", header = None) for item in filelist1], axis=1)

data2 = pd.concat([pd.read_table(dir2+'/'+item,delimiter="\n",sep=" ", header = None) for item in filelist2], axis=1)

print data1.shape
print data1.head(5)

print data2.shape
print data2.head(5)


