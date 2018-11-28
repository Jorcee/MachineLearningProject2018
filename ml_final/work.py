import fenci
import json


def cut(string=''):
    word = ''
    output = ''
    for c in string:
        if c.isalpha():
            word += c.lower()
        else:
            if word not in stop:
                if word:
                    output += p.stem(word, 0, len(word) - 1)
                output += c.lower()
            else:
                if word and word not in re_stop:
                    re_stop.append(word)
                    # print(word)
            word = ''
    return output


with open('pubs_train.json','r') as load_f:
    data_dict = json.load(load_f)
stop = []
re_stop = []
with open('stop.txt','r') as stop_f:
    for wd in stop_f.readlines():
        stop.append(str(wd[:-2]))
p = fenci.PorterStemmer()

for key in data_dict:
    for dict in data_dict[key]:
        dict["title"] = cut(dict["title"])
        if ("abstract" in dict) and dict["abstract"]:
            dict["abstract"] = cut(dict["abstract"])

with open("record.json", "w") as f:
    json.dump(data_dict, f)

# with open("stop.txt","w") as f:
#     for wd in re_stop:
#         print(wd)
#         f.write(wd)
#         f.write("\n")



