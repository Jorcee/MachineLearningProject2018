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
        n=len(data)
        correlation=numpy.zeros(shape=(n,n))
        # correlation matrix is a symmetric matrix
        for i in range(n):
            for j in range(i):
                correlation[i][j]=self.predict(data[i],data[j])
        correlation=correlation+correlation.T
        return correlation
