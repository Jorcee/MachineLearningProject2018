import random
class clf_abstract:
    
    def __init__(self):
        pass

    def train(self,data,label):
        print('clf_abstract train start')
        pass
        print('clf_abstract train end')

    def predict(self,paper1,paper2):
        return random.uniform(0,1)
