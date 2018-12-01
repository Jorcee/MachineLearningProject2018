import numpy
import random
import json
from bintree import *

def HierarchicalClustering(similarity,data):
    # n=len(data)
    # correlation=numpy.zeros(shape=(n,n))
    # # correlation matrix is a symmetric matrix
    # for i in range(n):
    #     for j in range(i):
    #         correlation[i][j]=similarity.predict(data[i],data[j])
    # correlation=correlation+correlation.T
    correlation = similarity.matrix(data)

    # #bruce-force
    # #total predicted pairs
    # tps=0
    # #correct predicted pairs, mathematical expectation of similarity function
    # cps=0
    # # p= cps/tps
    # p=0

    # # clusters has several lists
    # # each list is a cluster 
    # # use index in dataset to represent the paper
    # clusters=[[i] for i in range(n)]
    
    # while(1):
    #     m,n=-1,-1
    #     dp=0
    #     print(len(clusters))
    #     for i in range(len(clusters)):
    #         for j in range(i):
    #             dcps = correlation[clusters[i]][:,clusters[j]].sum()
    #             dtps = len(clusters[i]) * len(clusters[j])
    #             if ((dcps+cps)/(dtps+tps)-p)> dp:
    #                 m,n=i,j
    #                 dp=((dcps+cps)/(dtps+tps)-p)
    #     print(m,n)

    #     if(m<0):
    #         break
    #     cps=cps+dcps
    #     tps=tps+dtps
    #     p=p+dp
    #     print(cps,tps,p)
    #     clusters[m]=clusters[m]+clusters[n]
    #     del clusters[n]
    

    # nearst neighbor chain
    n = len(data) 
    list_old=[Node(i) for i in range(n)]
    list_new=[]
    clusteringtrees=[]
    numbers=n
    while(numbers>1):
        index=-1
        maxmum=0
        if(len(list_new)==0):
            list_new.append(list_old.pop())
        for j in range(len(list_old)):
            sim=node_simularity(correlation,list_new[-1],list_old[j])
            if(sim>maxmum):
                maxmum = sim
                index = j
        
        if (len(list_new)>1):
            sim=node_simularity(correlation,list_new[-1],list_new[-2])
            if (sim >= maxmum):
                if( sim > 0.55 ):
                #in_sim=(list_new[-1].similarity*list_new[-1].num+list_new[-2].similarity*list_new[-2].num)/(list_new[-1].num+list_new[-2].num)
                #if(sim>0.75*in_sim):
                    index1=min(list_new[-1].index,list_new[-2].index)
                    correlation[index1,:]=(list_new[-1].num*correlation[list_new[-1].index,:]+list_new[-2].num*correlation[list_new[-2].index,:])/(list_new[-1].num+list_new[-2].num)
                    correlation[:,index1]=(list_new[-1].num*correlation[:,list_new[-1].index]+list_new[-2].num*correlation[:,list_new[-2].index])/(list_new[-1].num+list_new[-2].num)
                    list_old.append(combine(list_new.pop(),list_new.pop(),sim))
                else:
                    clusteringtrees.append(list_new.pop())
                numbers-=1
            else:
                list_new.append(list_old.pop(index))
        else:
            list_new.append(list_old.pop(index))
        # print(len(list_new),len(list_old))
    clusteringtrees=clusteringtrees+list_new+list_old

    # for i in clusteringtrees:
    #     print(preorder(i))

    clusters = [preorder(i) for i in clusteringtrees]  
    return [ [data[i]['id'] for i in j] for j in clusters]

