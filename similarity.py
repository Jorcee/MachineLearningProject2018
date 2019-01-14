import random
import numpy
from sklearn.neural_network import MLPClassifier
class similarity:
    def __init__(self,*clfs):
        self.clfs=clfs
        self.trained=False
    def predict(self,paper1,paper2):
        probs_true=[]
        for clf in self.clfs:
            probs_true.append(clf.predict(paper1,paper2))
        
        p_true,p_false=1,1
        for p in probs_true:
            p_true *= p
            p_false *= (1-p)
        return p_true/(p_true+p_false)

    def train(self,data):
        self.mlp=MLPClassifier(activation='logistic',hidden_layer_sizes=(10,4), random_state=1,learning_rate_init=0.01,max_iter=100000,alpha=0.001)
        n=len(data)
        true_mat=numpy.ones(shape=(n,n))
        false_mat=numpy.ones(shape=(n,n))
        data_new=numpy.zeros(shape=(n**2,len(self.clfs)))
        label_new=numpy.zeros(shape=(n**2,len(self.clfs)))
        i=0
        for clf in self.clfs:
            ci=clf.matrix(data)
            for j in range(n):
                for k in range(n):
                    data_new[j*n+k,i]=ci[j,k]
                    if(data[j]['clus']==data[k]['clus']):
                        label_new[j*n+k]=1
                    else:
                        label_new[j*n+k]=0
            i+=1
        self.mlp.fit(data_new,label_new)
        self.trained=True


    def matrix(self,data):
        # prob_mats=[]
        # for clf in self.clfs:
        #     prob_mats.append(clf.matrix(data))
        # n=len(data)
        # true_mat=numpy.ones(shape=(n,n))
        # false_mat=numpy.ones(shape=(n,n))
        # for prob_mat in prob_mats:
        #     true_mat=true_mat*prob_mat
        #     false_mat=false_mat*(1-prob_mat)
        # correlation= true_mat/(true_mat+false_mat)
        # correlation[numpy.where((true_mat+false_mat)==0)]=1
        # return correlation
        if(self.trained):
            n=len(data)
            data_new=numpy.zeros(shape=(n**2,len(self.clfs)))
        
            i=0
            for clf in self.clfs:
                ci=clf.matrix(data)
                for j in range(n):
                    for k in range(n):
                        data_new[j*n+k,i]=ci[j,k]
                i+=1
            label_new=self.mlp.predict(data_new)

            return label_new.reshape((n,n),order='C')
        else:
            n=len(data)
            true_mat=numpy.ones(shape=(n,n))
            false_mat=numpy.ones(shape=(n,n))
            for clf in self.clfs:
                ci=clf.matrix(data)
                true_mat=true_mat*ci
                false_mat=false_mat*(1-ci)
            correlation=numpy.ones(shape=(n,n))
            for i in range(n):
                for j in range(n):
                    if(true_mat[i,j]+false_mat[i,j]==0):
                        correlation[i,j]=0.5
                    else:
                        correlation[i,j]=true_mat[i,j]/(true_mat[i,j]+false_mat[i,j])
            return correlation
        
        # n=len(data)
        # i=0
        # correlation=numpy.zeros(shape=(n,n))
        # for clf in self.clfs:
        #     ci=clf.matrix(data)
        #     correlation=correlation+ci
        #     i+=1

        # return correlation/i
        
        # n=len(data)
        # correlation=numpy.zeros(shape=(n,n))
        # # correlation matrix is a symmetric matrix
        # for i in range(n):
        #     for j in range(i):
        #         correlation[i][j]=self.predict(data[i],data[j])
        # correlation=correlation+correlation.T
        # return correlation
