import numpy
import re
import random
import matplotlib.pyplot as plt
class clf_org:

    # def org_preprocess(self, sentence):
    #     tags = []
    #     sentence_parts = sentence.lower().split()
    #     for i in range(0, len(sentence_parts)):
    #         for j in range(0, len(self.subjects)):
    #             if sentence_parts[i] == self.subjects[j]:
    #                 tags.append(self.subjects[j])
    #         for j in range(0, len(self.context_tags)):
    #             if sentence_parts[i] == self.context_tags[j]:
    #                 tags.append(sentence_parts[i + self.context_locs[j]])
    #     return tags

    def preprocess(self,name):
        return re.sub("[^A-Za-z]","",name).lower()

    def __init__(self):
        self.bayes=False
        self.N1=0
        self.N2=0
        self.kinds=101
        self.f1=numpy.zeros(self.kinds)
        self.f2=numpy.zeros(self.kinds)
        self.prob_bayes=numpy.zeros(self.kinds)
        self.max_iter=200
        # self.context_tags = ["institute", "academy", "university","univ"]
        # self.context_locs = [-1,-1,2]
        # self.subjects = ["chemistry", "biology", "physics", "math", "mathematics", "science"]

    def train(self,data,label):
        print('clf_org train start')
        self.bayes_train(data,label)
        print('clf_org train end')

    def predict(self,paper1,paper2):
        nameorg1=[self.preprocess(i['name']+i['org']) for i in paper1['authors'] ]
        nameorg2=[self.preprocess(i['name']+i['org']) for i in paper2['authors'] ]
        if(len(set(nameorg1)&set(nameorg2))>0):
            return 1
        else:
            return 0
        # for i in paper1['authors']:
        #     for j in paper2['authors']:
        #         if(self.preprocess(i['name'])==self.preprocess(j['name'])):
        #             if(self.preprocess(i['org'])==self.preprocess(j['org'])):
        #                 equal=1

        # return equal

        

    def matrix(self,data):
        n=len(data)
        correlation=numpy.zeros(shape=(n,n))
        # correlation matrix is a symmetric matrix
        if self.bayes:
            for i in range(n):
                for j in range(i):
                    correlation[i][j]=self.out2prob(self.predict(data[i],data[j]))
        else:
            for i in range(n):
                for j in range(i):
                    correlation[i][j]=self.predict(data[i],data[j])

        correlation=correlation+correlation.T
        return correlation


    def bayes_train(self,data,label):
        try:
            self.prob_bayes=numpy.load('clf_org_bayes.npy')
            print('load success')
        except IOError:

            iter=0
            names=list(label)
            random.shuffle(names)

            for name_ in names:
                
                data_=data[name_]
                label_=label[name_]
                len_=len(data_)
                if (len_>2000):
                    continue
            
                correlation = self.matrix(data_)
                #the propability-desity function of the out put of clf when real-correlation is 1
                # print(correlation)
                # the start and end of indexs of each clusters
                indexs=[len(i) for i in label_]

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
                    numpy.save('clf_org_bayes.npy',self.prob_bayes)
                    break
                self.prob_bayes = prob_bayes_new[:]
                numpy.save('clf_org_bayes.npy',self.prob_bayes)

        
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
        print('prob_bayes',self.prob_bayes)

    def out2prob(self,output):
        return self.prob_bayes[int(output*(self.kinds-1))]

