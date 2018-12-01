from gensim.models import KeyedVectors
import numpy as np
import re
import random
import matplotlib.pyplot as plt
class clf_keywords:
    
    def __init__(self):
        self.kv = KeyedVectors.load('wordvectors.kv')
        self.wv = KeyedVectors.load('wordvector2.kv')
        self.bayes=False
        self.N1=0
        self.N2=0
        self.kinds=101
        self.f1=np.zeros(self.kinds)
        self.f2=np.zeros(self.kinds)
        self.prob_bayes=np.zeros(self.kinds)
        self.max_iter=200
    def train(self,data,label):
        print('clf_keywords train start')
        self.bayes_train(data,label)
        print('clf_keywords train end')

    #Average of the average similarity of each keyword to every keyword in the other paper
    def predict(self,paper1,paper2):
        keys1 = paper1['keywords']
        keys2 = paper2['keywords']
        if not len(keys1) >= 1:
            return 0
        if not len(keys2) >= 1:
            return 0
        sim_values = []
        for word1 in keys1:
            temp = []
            for word2 in keys2:
                try:
                    a=self.kv.n_similarity(word1,word2)
                    if not np.isnan(a):
                        temp.append(a)
                except Exception:
                    try:
                        a=self.wv.n_similarity(word1,word2)
                        if not np.isnan(a):
                            temp.append(a)
                    except Exception:
                        temp = temp
            temp = np.array(temp)
            temp = np.mean(temp)
            sim_values.append(temp)
        sim_values = np.array(sim_values)
        return np.mean(sim_values)

    def matrix(self,data):
        n=len(data)
        correlation=np.zeros(shape=(n,n))
        # correlation matrix is a symmetric matrix
        for i in range(n):
            for j in range(i):
                #print(data[i],data[j])
                correlation[i][j]=self.predict(data[i],data[j])
        correlation=correlation+correlation.T
        return correlation


    def bayes_train(self,data,label):
        try:
            self.prob_bayes=np.load('clf_keywords_bayes.npy')
            print('load success')
        except IOError:

            iter=0
            names=list(label)
            random.shuffle(names)

            for name_ in names:
                data_=data[name_]
                label_=label[name_]
                len_=len(data_)
            
                correlation = self.matrix(data_)
                #the propability-desity function of the out put of clf when real-correlation is 1
                # print(correlation)
                # the start and end of indexs of each clusters
                indexs=[len(i) for i in label_]

                item=0
                for i in indexs:
                    item+=i
                len_=item

                dN1=0
                for len_clus in indexs:
                    dN1+=len_clus*(len_clus-1)/2 
                self.N1 += dN1
                self.N2 += (len_*(len_-1)/2 - dN1)

                for i in range(1,len(indexs)):
                    indexs[i]+=indexs[i-1]
                indexs=[0]+indexs
                # print(indexs)
                for i in range(len(indexs)-1):
                    for k in range(indexs[i],indexs[i+1]):
                        for j in range(k+1,indexs[i+1]):
                            self.f1[int(correlation[k][j]*(self.kinds-1))] += 1
                        for j in range(indexs[i+1],len_):
                            self.f2[int(correlation[k][j]*(self.kinds-1))] += 1
                prob_bayes_new=[]
                for i in range (self.kinds):
                    if(self.f1[i]+self.f2[i]<20):
                        #the result is invalueble, 2 is a label for smoothing
                        prob_bayes_new.append(2)
                    else:
                        #prob_bayes_new.append(self.f1[i]/(self.f1[i]+self.f2[i]))
                        prob_bayes_new.append(1*self.f1[i]/self.N1/(1*self.f1[i]/self.N1+1*self.f2[i]/self.N2))

                prob_bayes_new=self.smoothing(prob_bayes_new)

                iter+=1 
                print('learning step: {}',iter)
                if ((max([abs(prob_bayes_new[i]-self.prob_bayes[i]) for i in range(self.kinds)]) < 0.001 ) or (iter==self.max_iter)):
                    self.prob_bayes = prob_bayes_new[:]
                    np.save('clf_keywords_bayes.npy',self.prob_bayes)
                    break
                self.prob_bayes = prob_bayes_new[:]
                np.save('clf_keywords_bayes.npy',self.prob_bayes)

        
    def smoothing(self,prob):
        len_=len(prob)
        i=0
        while(i<len_):
            while(i<len_ and prob[i]<=1):
                i+=1
            if(i<len_):
                j=i+1
                while(j<len_ and prob[j]>1):
                    j+=1
                for k in range(i,j):
                    prob[k]=0
                    if (i>0):
                        prob[k]+=prob[i-1]* (j-k)/(j-i+1)
                    if (j<len_):
                        prob[k] += prob[j]* (k-i+1)/(j-i+1)
        return prob

    def plot(self):
        x=[k/(self.kinds-1) for k in range(self.kinds)]
        plt.plot(x,self.prob_bayes)
        plt.show()
    
    def print(self):
        print('N1',self.N1)
        print('N2',self.N2)
        print('prob',self.prob_bayes)
        print('f1',self.f1)
        print('f2',self.f2)
        if self.N1-sum(self.f1) !=0:
            print('asdfuawgefkagwekfhm')
        if self.N2-sum(self.f2) !=0:
            print('asdfuawgefkagwekfhm')

    def out2prob(self,output):
        return self.prob_bayes[int(output*(self.kinds-1))]
