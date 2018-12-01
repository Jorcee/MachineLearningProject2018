import numpy
import re
import random
from graph import graph
import matplotlib.pyplot as plt
class clf_coauthor:

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

    def train(self,data,label):
        print('clf_coauthor train start')
        self.bayes_train(data,label)
        print('clf_coauthor train end')

    def predict(self,paper1,paper2):
        namelist1=[i['name'] for i in paper1['authors']]
        namelist2=[i['name'] for i in paper2['authors']]
        if len(set(namelist1)&set(namelist2))>1:
            return 1
        else:
            return 0

    def matrix(self,data):
        n=len(data)
        authors_of_papers=[]
        authors_total={}
        for paper in data:
            authors_one_paper=[]
            for author in paper['authors']:
                name=self.preprocess(author['name'])
                authors_one_paper.append(name)
                if name in authors_total:
                    authors_total[name]+=1
                else:
                    authors_total[name]=1
            authors_of_papers.append(authors_one_paper)
        # find who is the author to be classified
        name_want=""
        m=0
        for name,times in authors_total.items():
            if(times>m):
                name_want=name
                m=times

        del authors_total[name_want]

        # add serial number
        index=0
        for name in authors_total:
            authors_total[name]=index+n
            index+=1

        #creat link matrix
        #link_mat = numpy.zeros(shape=(n+len(authors_total),n+len(authors_total)))


        N=[]
        for i in range(n+len(authors_total)):
            Ni=[]
            N.append(Ni)


        for i in range(n):
            indexs=[i]
            for author in authors_of_papers[i]:
                if (author != name_want):
                    indexs.append(authors_total[author])
            
            #print(indexs)
            for j in indexs:
                for k in indexs:
                    if(j!=k):
                        N[j].append(k)

        # calculate the mini-distance matrix

        graph1=graph(N)

        
        
        dis=graph1.distance(list(range(n)),list(range(n)))
        del graph1

        correlation=numpy.zeros(shape=(n,n))
        correlation[numpy.where(dis>0)]=1/numpy.sqrt(dis[numpy.where(dis>0)])
        # correlation matrix is a symmetric matrix
        if self.bayes:
            for i in range(n):
                for j in range(n):
                    correlation[i,j]=self.out2prob(correlation[i,j])

        return correlation



    def bayes_train(self,data,label):
        try:
            self.prob_bayes=numpy.load('clf_coauthor_bayes.npy')
            print('load success')
        except IOError:

            iter=0
            names=list(label)
            random.shuffle(names)

            for name_ in names:
                data_=data[name_]
                label_=label[name_]
                len_=len(data_)
                if (len_>1800):
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
                    numpy.save('clf_coauthor_bayes.npy',self.prob_bayes)
                    break
                self.prob_bayes = prob_bayes_new[:]
                numpy.save('clf_coauthor_bayes.npy',self.prob_bayes)
            


        
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

