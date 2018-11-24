context_tags = ["institute", "academy", "university"]
context_locs = [-1,-1,2]
subjects = ["chemistry", "biology", "physics", "math", "mathematics", "science"]
def preprocess(sentence):
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


def venue_A(venue_name):
    acronym = ""
    venue_parts = venue_name.split()
    for word in venue_parts:
        if word[0].isupper():
            acronym = acronym + word[0]
    return acronym
print(preprocess("Beijing Academy of Math and Science"))
print(preprocess("Department of Physics University of Georigie"))
print(venue_A("Journal of Liquid Chromotaogyaphy and Related Technollgies"))
