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


def venue_preprocess(venue_name):
    acronym = ""
    for word in venue_name.split():
        
        if word[0].isupper():
            acronym = acronym + word[0]
    return acronym
print(org_preprocess("Beijing Academy of Math and Science"))
print(preprocess("Department of Physics University of Georigie"))
print(venue_A("Journal of Liquid Chromotaogyaphy and Related Technollgies"))
