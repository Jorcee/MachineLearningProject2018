import numpy
import matplotlib.pyplot as plt
class bayes_tran:
    def __init__(self,clf,data,label):
        self.clf=clf
        self.N1=0
        self.N2=0
        self.kinds=101
        self.f1=numpy.zeros(self.kinds)
        self.f2=numpy.zeros(self.kinds)
        self.prob_bayes=numpy.zeros(self.kinds)
        for name_,label_ in label.items():
            data_=data[name_]
            len_=len(data_)
            correlation=numpy.zeros(shape=(len_,len_))
            for i in range(len_):
                print(i)
                for j in range(i):
                    correlation[i][j]=clf.predict(data_[i],data_[j])
                correlation[i][i]=0.5
            correlation=correlation+correlation.T
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
                # for k in range(indexs[i],indexs[i+1]):
                #     for j in range(indexs[i]):
                #         self.f2[int(correlation[k][j]*(self.kinds-1))] += 1
                #     for j in range(indexs[i],indexs[i+1]):
                #         self.f1[int(correlation[k][j]*(self.kinds-1))] += 1
                #     for j in range(indexs[i+1],len_):
                #         self.f2[int(correlation[k][j]*(self.kinds-1))] += 1
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

        
            if max([abs(prob_bayes_new[i]-self.prob_bayes[i]) for i in range(self.kinds)]) < 0.001:
                self.prob_bayes = prob_bayes_new[:]
                return
            self.prob_bayes = prob_bayes_new[:]

    def out2prob(self,output):
        return self.prob_bayes[int(output*(self.kinds-1))]

    def predict(self,paper1,paper2):
        return self.out2prob(self.clf.predict(paper1,paper2))

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
                        prob[k]+=prob[i-1]*(0.5**(k-i+1))
                    if (j<len_):
                        prob[k] += prob[j]*(0.5**(j-k))
        return prob

    def plot(self):
        x=[k/(self.kinds-1) for k in range(self.kinds)]
        plt.plot(x,self.prob_bayes)
        plt.show()
    
    def print(self):
        print('N1',self.N1)
        print('N2',self.N2)
        print('f1',self.f1)
        print('f2',self.f2)


        

