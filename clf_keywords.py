class clf_keywords:
    
    def __init__(self):
        pass

    def train(self,data,label):
        print('clf_keywords train start')
        pass
        print('clf_keywords train end')

    def predict(self,paper1,paper2):
        namelist1= paper1['keywords']
        namelist2= paper2['keywords']
        if len(set(namelist1)&set(namelist2))>0:
            return 1
        else:
            return 0