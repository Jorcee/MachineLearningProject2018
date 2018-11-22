import tensorflow as tf

class TextParser:

    def __init__(self, file_path, *vocab_size):
        self.file_path = file_path
        self.vocab_size = vocab_size or 5000
        with open(self.file_path) as f:
         self.words = tf.compat.as_str(f.read())
    def get_words(self):
        return self.words

tester = TextParser("testtext.txt")
print(TextParser.get_words())