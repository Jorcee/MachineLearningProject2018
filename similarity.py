import random
def simularity(paper1,paper2):
    if(paper1['venue']==paper2['venue']):
        return random.uniform(0.8,1)
    else:
        return 0