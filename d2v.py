from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import common_texts
from PaperJSONHandler import PaperJSON

text = PaperJSON('pubs_train.json')
data = list(text.get_abstracts_tagged())
model = Doc2Vec(vector_size = 50, min_count=2, epochs = 40, workers=4)
model.build_vocab(data)
model.train(data,total_examples=model.corpus_count,epochs=model.epochs)
model.save('doc2vec.model')
model.wv.save('doc2vec.kv')