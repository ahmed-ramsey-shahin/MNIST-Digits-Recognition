from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle


def one_hot(y):
    y_new = np.zeros((y.shape[0], y.max() + 1))
    y_new[np.arange(y.shape[0]), y] = 1
    return y_new


def init_params():
    w1 = np.random.rand(256, 784) - 0.5
    b1 = np.random.rand(256, 1) - 0.5

    w2 = np.random.rand(128, 256) - 0.5
    b2 = np.random.rand(128, 1) - 0.5

    w3 = np.random.rand(10, 128) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2, w3, b3


def ReLU(z):
    return np.maximum(z, 0)


def dReLU(z):
    return z > 0


def softmax(z):
    return np.exp(z) / sum(np.exp(z))


def forward_prop(w1, b1, w2, b2, w3, b3, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)

    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)

    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)

    return z1, a1, z2, a2, z3, a3


def backward_prop(z1, a1, z2, a2, a3, w2, w3, X, y):
    y = one_hot(y).T

    dz3 = a3 - y
    dw3 = (1 / m) * dz3.dot(a2.T)

    dz2 = w3.T.dot(dz3) * dReLU(z2)
    dw2 = (1 / m) * dz2.dot(a1.T)

    dz1 = w2.T.dot(dz2) * dReLU(z1)
    dw1 = (1 / m) * dz1.dot(X.T)

    return (1 / m) * np.sum(dz1), dw1, (1 / m) * np.sum(dz2), dw2, (1 / m) * np.sum(dz3), dw3


def update_w_b(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1

    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2

    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3

    return w1, b1, w2, b2, w3, b3


def get_predictions(a3):
    return np.argmax(a3, axis=0)


def calc_accuracy(predictions, y):
    return (np.sum(predictions == y) / y.shape[0]) * 100


def gradient_descent(X, y, iterations, alpha):
    w1, b1, w2, b2, w3, b3 = init_params()
    for i in range(1, iterations):
        z1, a1, z2, a2, z3, a3 = forward_prop(w1, b1, w2, b2, w3, b3, X)
        db1, dw1, db2, dw2, db3, dw3 = backward_prop(z1, a1, z2, a2, a3, w2, w3, X, y)
        w1, b1, w2, b2, w3, b3 = update_w_b(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)
        if (i + 1) % 10 == 0:
            predictions = get_predictions(a3)
            accuracy = calc_accuracy(predictions, y)
            print('Iteration {}, accuracy on the training set {}'.format(i + 1, accuracy))
    return w1, b1, w2, b2, w3, b3


# Reading the dataset files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Converting the dataset to numpy arrays
X_train = train.to_numpy()[:, 1:].T / 255
X_test = test.to_numpy()[:, 1:].T / 255
# Splitting the dataset into data and labels
y_train = train.to_numpy()[:, 0].T
y_test = test.to_numpy()[:, 0].T
# Defining some constants
m = X_train.shape[1]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()
idx = np.random.randint(0, X_train.shape[1], size=10)
for i in range(10):
    axes[i].imshow(X_train[:, idx[i]].reshape(28, 28), cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(str(int(y_train[idx[i]])), color='black', fontsize=25)
plt.show()

w1, b1, w2, b2, w3, b3 = gradient_descent(X_train, y_train, 500, 0.1)
print('Accuracy on the test set is: {}'.format(calc_accuracy(get_predictions(forward_prop(w1, b1, w2, b2, w3, b3, X_test)[-1]), y_test)))
file = open('model.obj', 'wb')
weights = {
    'w1': w1,
    'b1': b1,
    'w2': w2,
    'b2': b2,
    'w3': w3,
    'b3': b3
}
pickle.dump(weights, file)
file.close()
