import random
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import Doc2VecKeyedVectors
class clf_abstract: 
    def __init__(self):
        self.doc_vector = Doc2VecKeyedVectors.load('doc2vec.kv')

    def train(self,data,label):
        print('clf_abstract train start')
        pass
        print('clf_abstract train end')
    #expects paper1 and paper2 to be loaded json documents
    def predict(self,paper1, paper2):
        ab1 = simple_preprocess(paper1['abstract'])
        ab2 = simple_preprocess(paper1['abstract'])
        try:
            sim = self.doc_vector.n_similarity(ab1,ab2)
        except Exception:
            return 0
        return sim


