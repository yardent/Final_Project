import sys
import RNN_Algorithem
from datetime import datetime
import numpy


class PredicateSentenceWithSgd:
    """
    Predicate sentence with sgd algorithm.
    """
    def __init__(self, vocabulary_size, x_train, y_train, output_path, num_sentences, sentence_min_length, nepoch):
        numpy.random.seed(10)
        rnn_model = RNN_Algorithem.RNNNumpy(vocabulary_size)
        self.rnn_model = rnn_model
        self.x_train = x_train
        self.y_train = y_train
        self.output_path = output_path
        self.num_sentences = num_sentences
        self.sentence_min_length = sentence_min_length
        self.vocabulary_size = vocabulary_size
        self.learning_rate = 0.005
        self.nepoch = nepoch
        self.evaluate_loss_after = 1
        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"

    def train_with_sgd(self):
            # We keep track of the losses so we can plot them later
            losses = []
            num_examples_seen = 0
            learning_rate = self.learning_rate
            for epoch in range(self.nepoch):
                # Optionally evaluate the loss
                if epoch % self.evaluate_loss_after == 0:
                    loss = self.rnn_model.calculate_loss(self.x_train, self.y_train)
                    losses.append((num_examples_seen, loss))
                    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
                    # Adjust the learning rate if loss increases
                    if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                        learning_rate = learning_rate * 0.5
                        print("Setting learning rate to %f" % learning_rate)
                    sys.stdout.flush()
                # For each training example...
                for i in range(len(self.y_train)):
                    # One SGD step
                    self.rnn_model.sgd_step(self.x_train[i], self.y_train[i], self.learning_rate)
                    num_examples_seen += 1

    def create_predicate_sentence_file(self, word_to_index, index_to_word):
        text = ""
        for i in range(self.num_sentences):
            sent = []
            # We want long sentences, not sentences with one or two words
            while len(sent) < self.sentence_min_length:
                sent = self.generate_sentence(word_to_index, index_to_word)
            text = text + " ".join(sent) + "." + "\n"
        with open(self.output_path, "w") as text_file:
            text_file.write(text)

    def generate_sentence(self, dic_word_to_index, dic_index_to_word):
        # We start the sentence with the start token
        sentence_new = [dic_word_to_index[self.sentence_start_token]]
        # Repeat until we get an end token
        while not sentence_new[-1] == dic_word_to_index[self.sentence_end_token]:
            next_word_probs = self.rnn_model.forward_propagation(sentence_new)
            sampled_word = dic_word_to_index[self.unknown_token]
            # We don't want to sample unknown words
            while sampled_word == dic_word_to_index[self.unknown_token]:
                samples = numpy.random.multinomial(1, next_word_probs[0][-1])
                sampled_word = numpy.argmax(samples)
            sentence_new.append(sampled_word)
        sentence = [dic_index_to_word[x] for x in sentence_new[1:-1]]
        return sentence
