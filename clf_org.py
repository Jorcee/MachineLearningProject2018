class clf_org:
    context_tags = ["institute", "academy", "university","univ"]
    context_locs = [-1,-1,2]
    subjects = ["chemistry", "biology", "physics", "math", "mathematics", "science"]
    def org_preprocess(sentence):
        tags = []
        sentence_parts = sentence.lower().split()
        for i in range(0, len(sentence_parts)):
            for j in range(0, len(subjects)):
                if sentence_parts[i] == subjects[j]:
                    tags.append(subjects[j])
            for j in range(0, len(context_tags)):
                if sentence_parts[i] == context_tags[j]:
                    tags.append(sentence_parts[i + context_locs[j]])
        return tags

    def __init__(self):
        pass

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