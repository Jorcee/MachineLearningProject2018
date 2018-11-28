from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import PaperJSONHandler
text = PaperJSONHandler.PaperJSON('pubs_train.json')
abstracts = text.get_abstracts()
path = get_tmpfile('word2vec.model')
model = Word2Vec(abstracts, size=100, window=5, min_count=1, workers=4)
model.save('word2vec.model')
model.wv.save('wordvectors.kv')