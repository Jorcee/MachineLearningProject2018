class clf_org:
    def org_preprocess(self, sentence):
        tags = []
        sentence_parts = sentence.lower().split()
        for i in range(0, len(sentence_parts)):
            for j in range(0, len(self.subjects)):
                if sentence_parts[i] == self.subjects[j]:
                    tags.append(self.subjects[j])
            for j in range(0, len(self.context_tags)):
                if sentence_parts[i] == self.context_tags[j]:
                    tags.append(sentence_parts[i + self.context_locs[j]])
        return tags

    def __init__(self):
        self.context_tags = ["institute", "academy", "university","univ"]
        self.context_locs = [-1,-1,2]
        self.subjects = ["chemistry", "biology", "physics", "math", "mathematics", "science"]

    def train(self,data,label):
        print('clf_org train start')
        pass
        print('clf_org train end')

    def predict(self,paper1,paper2):
        namelist1=[i['name'] for i in paper1['authors']]
        namelist2=[i['name'] for i in paper2['authors']]
        if len(set(namelist1)&set(namelist2))>1:
            return 1
        else:
            return 0