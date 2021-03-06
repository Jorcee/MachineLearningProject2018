import random
import numpy
class clf_ideal:
    def __init__(self):
        pass
    def train(self,data,label):
        print('clf_ideal train start')
        pass
        print('clf_ideal train end')
    def predict(self,paper1,paper2):
        if(paper1['clus']==paper2['clus']):
            return random.uniform(0.4,1.0)
        else:
            return random.uniform(0.0,0.6)

    def matrix(self,data):
        n=len(data)
        correlation=numpy.zeros(shape=(n,n))
        # correlation matrix is a symmetric matrix
        for i in range(n):
            for j in range(i):
                correlation[i][j]=self.predict(data[i],data[j])
        correlation=correlation+correlation.T
        return correlation