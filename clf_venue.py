def preprocess(venue_name):
        acronym = ""
        for word in venue_name.split():
            acronym+=word[0].lower()
        return acronym

class clf_venue:
    def __init__(self):
        pass
    
    def train(self,data,label):
        print('clf_venue train start')
        pass
        print('clf_venue train end')

    def predict(self,paper1,paper2):
        if preprocess(paper1['venue'])==preprocess(paper2['venue']):
            return 1
        else:
            return 0
            