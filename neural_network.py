import numpy as np
import pickle


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sigmoid_derv(s):
    return 1. * (s > 0)


def cross_entropy(predicted, real):
    return (predicted - real) / real.shape[0]


def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss


class NeuralNetwork:
    def __init__(self, x_train, y_train, num_of_neurons):
        self.x_train = x_train
        self.y_train = y_train
        self.num_of_neurons = num_of_neurons
        self.learning_rate = 0.1
        input_dim = x_train.shape[1]
        output_dim = y_train.shape[1]

        self.w1 = np.random.randn(input_dim, num_of_neurons)
        self.b1 = np.zeros((1, num_of_neurons))
        self.w2 = np.random.randn(num_of_neurons, num_of_neurons)
        self.b2 = np.zeros((1, num_of_neurons))
        self.w3 = np.random.randn(num_of_neurons, output_dim)
        self.b3 = np.zeros((1, output_dim))

        self.a1 = None
        self.a2 = None
        self.a3 = None

    def feed_forward(self):
        z1 = np.dot(self.x_train, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = sigmoid(z3)

    def back_prop(self):
        a3_delta = cross_entropy(self.a3, self.y_train)  # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2)  # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1)  # w1

        self.w3 -= self.learning_rate * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.learning_rate * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.learning_rate * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.learning_rate * np.sum(a2_delta, axis=0)
        self.w1 -= self.learning_rate * np.dot(self.x_train.T, a1_delta)
        self.b1 -= self.learning_rate * np.sum(a1_delta, axis=0)

    def train(self, num_of_epochs=10, learning_rate=0.1):
        self.learning_rate = learning_rate
        for i in range(num_of_epochs):
            self.feed_forward()
            self.back_prop()
            print("Epoch: ", i, "loss: ", error(self.a3, self.y_train))

    def predict(self, data):
        self.x_train = data
        self.feed_forward()
        return self.a3.argmax()

    def load_weights(self, path):
        try:
            with open(path, 'rb') as file:
                (self.w1, self.w2, self.w3, self.b1, self.b2, self.b3) = pickle.load(file)
        finally:
            print("Could not load file")
            pass

    def save_weights(self, path):
        try:
            with open(path, 'wb') as file:
                pickle.dump((self.w1, self.w2, self.w3, self.b1, self.b2, self.b3), file, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            print("Could not save file")
            pass
