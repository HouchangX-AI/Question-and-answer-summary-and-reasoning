from utils.data_utils import load_list
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re


class Feature(object):

    def __init__(self, data=None,
                 feature_type='tfidf_char',
                 feature_vec_path=None,
                 is_infer=False,
                 min_count=1,
                 word_vocab=None,
                 max_len=400):
        self.data_set = data
        self.feature_type = feature_type
        self.feature_vec_path = feature_vec_path
        self.sentence_symbol = load_list(path='datasets/sentence_symbol.txt')
        self.stop_words = load_list(path='datasets/stop_words.txt')
        self.is_infer = is_infer
        self.min_count = min_count
        self.word_vocab = word_vocab
        self.max_len = max_len

    def get_feature(self):
        if self.feature_type == 'vectorize':
            data_feature = self.vectorize(self.data_set)
        elif self.feature_type == 'doc_vectorize':
            data_feature = self.doc_vectorize(self.data_set)

        return data_feature

    def vectorize(self, data_set):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data_set)
        sequences = tokenizer.fit_on_sequences(data_set)

        # word_index = tokenizer.word_index
        data_feature = pad_sequences(sequences, maxlen=self.max_len)

        return data_feature

    def doc_vectorize(self, data_set, max_sentences=16):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data_set)

        data_feature = np.zeros((len(data_set), max_sentences, self.max_len), dtype='int32')
        for i, sentence in enumerate(data_set):
            sentence_symbols = "".join(self.sentence_symbol)
            split = "[" + sentence_symbols + "]"
            short_sents = re.split(split, sentence)
            for j, sent in enumerate(short_sents):
                if j < max_sentences and sent.strip():
                    words = text_to_word_sequence(sent)
                    k = 0
                    for w in words:
                        if k < self.max_len:
                            if w in tokenizer.word_index:
                                data_feature[i, j, k] = tokenizer.word_index[w]
                            k += 1
        # word_index = tokenizer.word_index
        return data_feature


