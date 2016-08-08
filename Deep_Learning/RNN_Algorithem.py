import numpy
import operator


class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.npa = numpy.array
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = numpy.random.uniform(-numpy.sqrt(1. / word_dim), numpy.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = numpy.random.uniform(-numpy.sqrt(1. / hidden_dim), numpy.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = numpy.random.uniform(-numpy.sqrt(1. / hidden_dim), numpy.sqrt(1. / hidden_dim),
                                      (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        t = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = numpy.zeros((t + 1, self.hidden_dim))
        s[-1] = numpy.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = numpy.zeros((t, self.word_dim))
        # For each time step...
        for t in numpy.arange(t):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = numpy.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = self.softmax(self.V.dot(s[t]))

        return [o, s]

        # RNNNumpy.forward_propagation = forward_propagation

    def softmax(self, w, t=1.0):
        e = numpy.exp(self.npa(w) / t)
        dist = e / numpy.sum(e)
        return dist
        # RNNNumpy.softmax = softmax

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return numpy.argmax(o, axis=1)

        # RNNNumpy.predict = predict

    def calculate_total_loss(self, x, y):
        l = 0
        # For each sentence...
        for i in numpy.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[numpy.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            l += -1 * numpy.sum(numpy.log(correct_word_predictions))
        return l

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        n = numpy.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / n

    def bptt(self, x, y):
        t = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        d_l_d_u = numpy.zeros(self.U.shape)
        d_l_d_v = numpy.zeros(self.V.shape)
        d_l_d_w = numpy.zeros(self.W.shape)
        delta_o = o
        delta_o[numpy.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in numpy.arange(t)[::-1]:
            d_l_d_v += numpy.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in numpy.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                d_l_d_w += numpy.outer(delta_t, s[bptt_step - 1])
                d_l_d_u[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [d_l_d_u, d_l_d_v, d_l_d_w]

    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        d_l_d_u, d_l_d_v, d_l_d_w = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * d_l_d_u
        self.V -= learning_rate * d_l_d_v
        self.W -= learning_rate * d_l_d_w
