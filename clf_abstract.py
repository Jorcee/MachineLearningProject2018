import random
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import Doc2VecKeyedVectors
from gensim.models import Doc2Vec
from scipy import spatial
import numpy
class clf_abstract: 
    def __init__(self):
        self.doc_vector = Doc2VecKeyedVectors.load('doc2vec.kv')
        self.doc_model = Doc2Vec.load('doc2vec.model')

    def train(self,data,label):
        print('clf_abstract train start')
        pass
        print('clf_abstract train end')
    #expects paper1 and paper2 to be vectors fro the abstracts of those papers
    def predict(self,paper1, paper2):
        return 1 - spatial.distance.cosine(paper1, paper2)

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
        doc = simple_preprocess(paper['abstract'])
        try:
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
        for i in range(n):
            print(i)
            for j in range(i):
                correlation[i][j]=spatial.distance.cosine(vecs[i],vecs[j])
                #
                if (correlation[i][j]>1):
                    print(correlation[i][j])
                #
        correlation=correlation+correlation.T
        return correlation

