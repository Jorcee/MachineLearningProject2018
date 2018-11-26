import json
from score import*
from clustering import*

with open('pubs_train.json', 'r') as f:
    data = json.load(f)

with open('assignment_train.json') as f:
    label = json.load(f)

for author in label:
    data[author],label[author]
    a = score(HierarchicalClustering(data[author]),label[author],'precise')
    print(a)