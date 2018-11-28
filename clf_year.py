import random
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
        diff = float(diff) * 0.05
        return 1 - diff
        