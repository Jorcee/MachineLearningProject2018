import json

class PaperJSON:
    def __init__(self,file_name):
        with open(file_name) as file:
            self.json = json.load(file)
    def get_abstracts(self):
        abstracts = []
        for author in self.json:
            for paper in self.json[author]:
                try:
                    if type(paper['abstract']) == str:
                        abstracts.append(paper['abstract'])
                except Exception:
                    filler = 1
        return abstracts