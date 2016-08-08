import os
import glob
import nltk
import string
import re
import itertools
import numpy
import RNN_Algorithem


class Preprocess:

    def __init__(self, vocabulary_size):
        # Assign instance variables
        self.vocabulary_size = vocabulary_size
        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"
        self.X_train = None
        self.y_train = None
        self.dic_index_to_word = None
        self.dic_word_to_index = None

    def get_train_set(self, words):
        word_freq = nltk.FreqDist(itertools.chain(*words))
        # -1 because unknown token
        vocab = word_freq.most_common(self.vocabulary_size - 1)
        dic_index_to_word = [x[0] for x in vocab]
        # adding unknown token
        dic_index_to_word.append(self.unknown_token)
        # filling
        dic_word_to_index = dict([(w, i) for i, w in enumerate(dic_index_to_word)])
        self.dic_index_to_word = dic_index_to_word
        self.dic_word_to_index = dic_word_to_index
        for i, sent in enumerate(words):
            words[i] = [w if w in dic_word_to_index else self.unknown_token for w in sent]
        # split to two arrays, one with start sentence and one with end sentence.
        self.X_train = numpy.asarray([[dic_word_to_index[w] for w in sent[:-1]] for sent in words])
        self.y_train = numpy.asarray([[dic_word_to_index[w] for w in sent[1:]] for sent in words])

    def split_sentence(self, content):
        sentences = content.split('\n')
        # sentences = sentences.split()
        # sentences = sentences.split()
        self.split_words(sentences)

    def do_preprocess(self, path):
        with open(path, 'r') as file_content:
            content_str = file_content.read()
        content_str = content_str.lower()
        self.split_sentence(content_str)
        return [self.X_train, self.y_train, self.vocabulary_size, self.dic_word_to_index, self.dic_index_to_word]

    def split_words(self, sentences):
        sentences = [re.sub('[%s]' % re.escape(string.punctuation), '', sent) for sent in sentences]
        sentences = [sent.replace('\n', ' ') for sent in sentences]
        sentences = self.add_start_end_token(sentences)
        words = [nltk.word_tokenize(sent) for sent in sentences]
        self.get_train_set(words)

    def add_start_end_token(self, sentences):
        sentences = ["%s %s %s" % (self.sentence_start_token, x, self.sentence_end_token) for x in sentences]
        return sentences
