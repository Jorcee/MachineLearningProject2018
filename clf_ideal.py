import random
class clf_ideal:
    def __init__(self):
        pass
    def train(self,data,label):
        print('clf_ideal train start')
        pass
        print('clf_ideal train end')
    def predict(self,paper1,paper2):
        if(paper1['clus']==paper2['clus']):
            return random.uniform(0.2,1.0)
        else:
            return random.uniform(0.0,0.8)