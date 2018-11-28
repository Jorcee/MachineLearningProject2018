from gensim.models import KeyedVectors
import numpy as np
class clf_keywords:
    
    def __init__(self):
        self.kv = KeyedVectors.load('wordvectors.kv')

    def train(self,data,label):
        print('clf_keywords train start')
        pass
        print('clf_keywords train end')

    #Average of the average similarity of each keyword to every keyword in the other paper
    def predict(self,paper1,paper2):
        keys1 = paper1['keywords']
        keys2 = paper2['keywords']
        sim_values = []
        for word1 in keys1:
            temp = []
            for word2 in keys2:
                try:
                    temp.append(self.kv.similarity(word1,word2))
                except Exception:
                    temp.append(0)
            temp = np.array(temp)
            temp = np.mean(temp)
            sim_values.append(temp)
        sim_values = np.array(sim_values)
        return np.mean(sim_values)

