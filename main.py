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


# clf1=clf_venue()
# clf1.train(data,label)
# clf1_new= bayes_tran(clf1,data,label)

# clf2=clf_org()
# clf2.train(data,label)
# clf2_new= bayes_tran(clf2,data,label)

# clf3=clf_keywords()
# clf3.train(data,label)
# clf3_new= bayes_tran(clf3,data,label)

# similarity_total=similarity(clf1_new,clf2_new,clf3_new)

clf=clf_abstract()
clf_new= bayes_tran(clf, data,label)

similarity_total=similarity(clf_new)

for name in label:
    data[name],label[name]
    a = score(HierarchicalClustering(similarity_total,data[name]),label[name],'F1')
    print(a)

clf_new.plot()