import gensim
import logging
import TextHandler
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
text = TextHandler.TextHandler('testtext.txt')
sentences = text.sentences
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)
print(model.wv['the'])
