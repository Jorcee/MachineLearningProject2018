import json
from score import*
from clustering import*

f=open('lima.dat','r')
data_lima=json.load(f)
g=open('lima.lab','r')
clus_lima=json.load(g)

print(score(Max_Pair_Clus(data_lima),clus_lima,'precise'))
