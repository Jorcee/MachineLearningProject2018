from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import get_tmpfile

class SenSim:
    def __init__(self):
        self.word_vectors = KeyedVectors.load('wordvectors.kv', mmap='r')
    def predict(self, paper1, paper2):
        return self.word_vectors.wmdistance(paper1['abstract'], paper2['abstract'])