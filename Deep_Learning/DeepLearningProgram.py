import os
import glob
import Preprocess
import RNN_Algorithem
import numpy
import PredicateSentenceWithSgd

# Manage all deep learning program

# Run configuration
path = "/home/mapr/PycharmProjects/Deep_Learning/data/the-wolf-of-wall-street-fix-no-low.txt"
output_path = "/home/mapr/PycharmProjects/Deep_Learning/results/Output-oran-last-no-low-new.txt"
learning_rate = 0.005
nepoch = 50
num_sentences = 100
senten_min_length = 3
vocabulary_size = 2000

# step 1: preprocess
model_preprocess = Preprocess.Preprocess(vocabulary_size)
X_train, y_train, vocabulary_size, word_to_index, index_to_word = model_preprocess.do_preprocess(path)

# step 2: train sgd
model_classifier = PredicateSentenceWithSgd.PredicateSentenceWithSgd(vocabulary_size, X_train, y_train, output_path,
                                                                     num_sentences, senten_min_length, nepoch)
losses = model_classifier.train_with_sgd()

# step 3: predict sentences
model_classifier.create_predicate_sentence_file(word_to_index, index_to_word)
