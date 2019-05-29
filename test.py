import pickle
import numpy as np
import neural_network


def to_hot(labels, num_of_categories):
    res = np.zeros(shape=(labels.shape[0], num_of_categories))
    for i in range(0, len(labels)):
        res[i, labels[i]] = 1
    return np.array(res)


def normalize(x):
    x = np.array(x)
    x[np.abs(x) < 0.35] = 0
    x_mean = np.mean(x, axis=0)
    x -= x_mean
    return x


def divide_train_test(x, y, ratio):
    x_test = x[int(x.shape[0] * ratio):]
    x_train = x[:int(x.shape[0] * ratio)]
    y_test = y[int(y.shape[0] * ratio):]
    y_train = y[:int(y.shape[0] * ratio)]
    return x_train, y_train, x_test, y_test


def test():
    with open('cropped.pkl', 'rb') as file:
        x_train, y_train = pickle.load(file)
    x_train = np.reshape(x_train, (x_train.shape[0], 784))
    x_train = normalize(x_train)
    y_train = to_hot(y_train, 10)
    x_train, y_train, x_test, y_test = divide_train_test(x_train, y_train, 0.8)

    model = neural_network.NeuralNetwork(x_train, y_train, 36)
    model.train(1000, 1)
    correct = 0
    for i in range(y_test.shape[0]):
        result = model.predict(x_test[i])
        if result == np.argmax(y_test[i]):
            correct += 1
    print(correct/y_test.shape[0])


if __name__ == "__main__":
    test()
