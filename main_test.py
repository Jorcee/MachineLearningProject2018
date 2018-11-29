import json
from score import*
from clustering import*
from similarity import *
from bayes_tran import*
from clf_ideal import clf_ideal
from clf_venue import clf_venue
from clf_org import clf_org
from clf_keywords import clf_keywords
from clf_abstract import clf_abstract
from clf_year import clf_year
from clf_coauthor import clf_coauthor
import re 
def preprocess(name):
    return re.sub("[^A-Za-z]","",name).lower()

MODE = 'test'
#MODE = 'run'
if MODE == 'run':
    with open('pubs_train.json', 'r') as f:
        data = json.load(f)

    with open('assignment_train.json') as f:
        label = json.load(f)

    for data_item in data.values():
        for data_single in data_item:
            if 'abstract' not in data_single:
                data_single['abstract']=''

    # add label for calculate a=b fastly in training step
    for name,label_item in label.items():
        index=0
        for i in range(len(label_item)):
            for j in range(len(label_item[i])):
                data[name][index]['clus']=i
                index+=1
else:
    with open('lima.dat','r') as f:
        data_lima=json.load(f)
    with open('lima.lab','r') as g:
        label_lima=json.load(g)
    for data_item in data_lima:
        if 'abstract' not in data_item:
            data_item['abstract']=''

    index=0
    for i in range(len(label_lima)):
        for j in range(len(label_lima[i])):
            data_lima[index]['clus']=i
            index+=1

    data={'li_ma':data_lima}
    label={'li_ma':label_lima}



index=0
has_lima=0
for i in range(len(label_lima)):
    for j in range(len(label_lima[i])):
        authors=data_lima[index]['authors']
        k=0
        for author in authors:
            filter(str.isalpha, (author['name']))
            if preprocess(author['name'])=='lima':
                k=1
        if k==1:
            has_lima+=1
        #else:
            #print(authors)
        index+=1

print(has_lima,index)

clf3=clf_year()
clf3.train(data,label)
clf3_new= bayes_tran(clf3,data,label)


clf3=clf_coauthor()
clf3.train(data,label)
clf3_new= bayes_tran(clf3,data,label)

similarity_total=similarity(clf3_new)


for name in label:
    data[name],label[name]
    label_=HierarchicalClustering(similarity_total,data[name])
    a = score(label_,label[name],'precise')
    b=score(label_,label[name],'recall')
    c=score(label_,label[name],'F1')
    print(a,b,c)

clf3_new.plot()
