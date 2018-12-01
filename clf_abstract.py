import random
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import Doc2VecKeyedVectors
from gensim.models import KeyedVectors
from gensim.models import Doc2Vec
from scipy import spatial
import numpy
import matplotlib.pyplot as plt
class clf_abstract: 
    def __init__(self):
        self.doc_vector = Doc2VecKeyedVectors.load('doc2vec.kv')
        self.doc_model = Doc2Vec.load('doc2vec.model')
        self.word_vector = KeyedVectors.load('wordvectors.kv')
        self.bayes=False
        self.N1=0
        self.N2=0
        self.kinds=101
        self.f1=numpy.zeros(self.kinds)
        self.f2=numpy.zeros(self.kinds)
        self.prob_bayes=numpy.zeros(self.kinds)
        self.max_iter=200

    def train(self,data,label):
        print('clf_abstract train start')
        self.bayes_train(data,label)
        print('clf_abstract train end')
    #expects paper1 and paper2 to be vectors fro the abstracts of those papers
    def predict(self,vec1, vec2):
        sim = 1 - spatial.distance.cosine(vec1, vec2)
        sim = max(min(1,sim),0)
        return sim

        #The old, crappy way
        # ab1 = simple_preprocess(paper1['abstract'])
        # ab2 = simple_preprocess(paper1['abstract'])
        # try:
        #     sim = self.doc_vector.n_similarity(ab1,ab2)
        # except Exception as e:
        #     print(e)
        #     return 0
        # return sim

    #Returns the vector of a paper's abstract
    def get_vector(self, paper):
        try:
            doc = simple_preprocess(paper['title'])
            return self.doc_model.infer_vector(doc)
        except Exception:
            return [0]

    def matrix(self,data):
        n=len(data)
        vecs=[]
        for paper in data:
            vecs.append(self.get_vector(paper))

        correlation=numpy.zeros(shape=(n,n))
        # correlation matrix is a symmetric matrix
        if self.bayes:
            for i in range(n):
                for j in range(i):
                    correlation[i][j]=self.out2prob(self.predict(vecs[i],vecs[j]))
        else:
            for i in range(n):
                for j in range(i):
                    correlation[i][j]=self.predict(vecs[i],vecs[j])
        correlation=correlation+correlation.T
        return correlation


    def bayes_train(self,data,label):
        try:
            self.prob_bayes=numpy.load('clf_abstract_bayes.npy')
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
                print(self.f1)
                if ((max([abs(prob_bayes_new[i]-self.prob_bayes[i]) for i in range(self.kinds)]) < 0.001 ) or (iter==self.max_iter)):
                    self.prob_bayes = prob_bayes_new[:]
                    numpy.save('clf_abstract_bayes.npy',self.prob_bayes)
                    break
                self.prob_bayes = prob_bayes_new[:]
                numpy.save('clf_abstract_bayes.npy',self.prob_bayes)
            


        
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

