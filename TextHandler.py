import tensorflow as tf
import collections
import numpy as np

#Reads a file and handles all the stuff involving the text
class TextHandler:

    def __init__(self, file_path):
        self.file_path = file_path
        self.text = []
        #Create a list of all the words in the text
        with open(self.file_path) as f:
            for word in tf.compat.as_str(f.read()).split():
                if word != '.':
                    self.text.append(word.lower())    
        self.make_sentence_list(file_path)
        self.make_vocab(self.text)
        self.make_dict()
        self.make_windows()
        self.make_train_test()

    #Make a list of sentences as lists
    def make_sentence_list(self, file_path):
        with open(self.file_path) as f:
            raw_sentences = f.read().split('.')
            self.sentences = []
            for sentence in raw_sentences:
                sentence = sentence.lower()
                self.sentences.append(sentence.split())

    #Makes a set of all the words
    def make_vocab(self, text):
        self.vocab = set(self.text)
        self.vocab_size = len(self.vocab)

    #Makes a dictionary
    def make_dict(self):
        self.word2int = {}
        self.int2word = {}
        for i, word in enumerate(self.vocab):
            self.word2int[word] = i
            self.int2word[i] = word
        return self.word2int, self.int2word
    
    #Get the training windows
    #A list of word pairs
    def make_windows(self):
        window_size = 2
        self.windows = []
        for sentence in self.sentences:
            for word_index, word in enumerate(sentence):
                for neighbor in sentence[max(word_index - window_size, 0) : min(word_index + window_size, len(sentence)) + 1]:
                    if neighbor != word:
                        self.windows.append([word, neighbor])
        return self.windows

    #Makes a One Hot Vector
    def make_one_hot(self, index, vocab_size = None):
        vocab_size = vocab_size or self.vocab_size
        temp = np.zeros(vocab_size)
        temp[index] = 1
        return temp
    
    #Makes a training set
    def make_train_test(self): 
        self.x_train = []
        self.y_train = []
        for word_pair in self.windows:
            self.x_train.append(self.make_one_hot(self.word2int[word_pair[0]]))
            self.y_train.append(self.make_one_hot(self.word2int[word_pair[1]]))
        self.x_train = np.asarray(self.x_train)
        self.y_train = np.asarray(self.y_train)
        return self.x_train, self.y_train