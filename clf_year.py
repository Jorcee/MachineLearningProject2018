import random
import numpy
class clf_year:
    
    def __init__(self):
        pass

    def train(self,data,label):
        print('clf_year train start')
        pass
        print('clf_year train end')

    #Same year = 1
    #Every year takes .05 away
    def predict(self,paper1,paper2):
        year1 = int(paper1['year'])
        year2 = int(paper2['year'])
        diff = abs(year1 - year2)
        diff = 2.718**(-diff*diff/100)
        return diff
    
    def matrix(self,data):
        n=len(data)
        correlation=numpy.zeros(shape=(n,n))
        # correlation matrix is a symmetric matrix
        for i in range(n):
            for j in range(i):
                correlation[i][j]=self.predict(data[i],data[j])
        correlation=correlation+correlation.T
        return correlation