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

f=open('lima.dat','r')
data_lima=json.load(f)
g=open('lima.lab','r')
label_lima=json.load(g)

index=0
for i in range(len(label_lima)):
    for j in range(len(label_lima[i])):
        data_lima[index]['clus']=i
        index+=1

data={'lima':data_lima}
label={'lima':label_lima}

clf1=clf_venue()
clf1.train(data,label)
clf1_new= bayes_tran(clf1,data,label)

clf2=clf_org()
clf2.train(data,label)
clf2_new= bayes_tran(clf2,data,label)

clf3=clf_keywords()
clf3.train(data,label)
clf3_new= bayes_tran(clf3,data,label)

similarity_total=similarity(clf1_new,clf2_new,clf3_new)

for name in label:
    data[name],label[name]
    a = score(HierarchicalClustering(similarity_total,data[name]),label[name],'F1')
    print(a)

