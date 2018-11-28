import json
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

class PaperJSON:
    def __init__(self,file_name):
        with open(file_name) as file:
            self.json = json.load(file)
    def get_abstracts_word2vec(self):
        abstracts = []
        for author in self.json:
            for paper in self.json[author]:
                try:
                    if type(paper['abstract']) == str:
                        abstracts.append(paper['abstract'])
                except Exception:
                    filler = 1
        return abstracts

    def get_abstracts_tagged(self):
        for author in self.json:
            for paper in self.json[author]:
                try:
                    if type(paper['abstract']) == str:
                        if type(paper['id']) == str:
                            yield TaggedDocument(simple_preprocess(paper['abstract']), paper['id'])
                except Exception:
                    filler = 1