import TextHandler
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import collections
class W2V:
    def __init__(self, file_path, num_iters = 1000):
        text = TextHandler.TextHandler(file_path)

        #Make placeholders for x and y
        x = tf.placeholder(tf.float32, shape=(None, text.vocab_size))
        y = tf.placeholder(tf.float32, shape=(None, text.vocab_size))

        #Dimensions to embed to
        emb_dim = 5

        W1 = tf.Variable(tf.random_normal([text.vocab_size, emb_dim]))
        b1 = tf.Variable(tf.random_normal([emb_dim])) #bias
        hidden = tf.add(tf.matmul(x,W1), b1)

        W2 = tf.Variable(tf.random_normal([emb_dim, text.vocab_size]))
        b2 = tf.Variable(tf.random_normal([text.vocab_size]))
        prediction = tf.nn.softmax(tf.add( tf.matmul(hidden, W2), b2))

        #training
        sess = tf.Session()

        init = tf.global_variables_initializer()
        sess.run(init)

        #loss function
        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),reduction_indices=[1]))

        #training step
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)


        #train it
        for _ in range(num_iters):
            sess.run(train_step, feed_dict={x: text.x_train, y: text.y_train})
            print("loss is: ", sess.run(cross_entropy_loss, feed_dict={x: text.x_train, y: text.y_train}))

        self.vectors = sess.run(W1 + b1)
        self.text = text

    #Euclidean Distance
    def e_dist(self, vec1, vec2):
        return np.sqrt(np.sum((vec1-vec2)**2))

    #Closest Word to the Other
    def find_closest(self, word):
        min_dist = 10000 # to act like positive infinity
        min_index = -1
        word_index = self.text.word2int[word.lower()]
        query_vector = self.vectors[word_index]
        for index, vector in enumerate(self.vectors):
            if self.e_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
                min_dist = self.e_dist(vector, query_vector)
                min_index = index
        return self.text.int2word[min_index]
    
    def build_dataset(self, words, n_words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary

test = W2V("testtext.txt", 50)
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(test.vectors)
normalizer = preprocessing.Normalizer()
vectors =  normalizer.fit_transform(vectors, 'l2')
fig, ax = plt.subplots()
for word in test.text.vocab:
    print(word, vectors[test.text.word2int[word]][1])
    ax.annotate(word, (vectors[test.text.word2int[word]][0],vectors[test.text.word2int[word]][1] ))
plt.show()