import numpy
class clf_venue:
    def __init__(self):
        pass
    def preprocess(self,venue_name):
        acronym = ""
        for word in venue_name.split():
            acronym+=word[0].lower()
        return acronym
    def train(self,data,label):
        print('clf_venue train start')
        pass
        print('clf_venue train end')

    def predict(self,paper1,paper2):
        if self.preprocess(paper1['venue'])==self.preprocess(paper2['venue']):
            return 1
        else:
            return 0

    def matrix(self,data):
        n=len(data)
        correlation=numpy.zeros(shape=(n,n))
        # correlation matrix is a symmetric matrix
        for i in range(n):
            for j in range(i):
                correlation[i][j]=self.predict(data[i],data[j])
        correlation=correlation+correlation.T
        return correlation
            