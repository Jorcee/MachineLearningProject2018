import random
import numpy
class similarity:
    def __init__(self,*clfs):
        self.clfs=clfs
    def predict(self,paper1,paper2):
        probs_true=[]
        for clf in self.clfs:
            probs_true.append(clf.predict(paper1,paper2))
        
        p_true,p_false=1,1
        for p in probs_true:
            p_true *= p
            p_false *= (1-p)
        return p_true/(p_true+p_false)

    def matrix(self,data):
        prob_mats=[]
        for clf in self.clfs:
            prob_mats.append(clf.matrix(data))
        n=len(data)
        true_mat=numpy.ones(shape=(n,n))
        false_mat=numpy.ones(shape=(n,n))
        for prob_mat in prob_mats:
            true_mat=true_mat*prob_mat
            false_mat=false_mat*(1-prob_mat)
        correlation= true_mat/(true_mat+false_mat)
        return correlation
        
        # n=len(data)
        # correlation=numpy.zeros(shape=(n,n))
        # # correlation matrix is a symmetric matrix
        # for i in range(n):
        #     for j in range(i):
        #         correlation[i][j]=self.predict(data[i],data[j])
        # correlation=correlation+correlation.T
        # return correlation
