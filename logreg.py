import gzip
import cPickle as pickle
import theano
import theano.tensor as T
import numpy as np
import lasagne

def load_mnist(path):

    tr, va, te = pickle.load(gzip.open(path, 'r'))

    tr_x, tr_y = tr
    va_x, va_y = va
    te_x, te_y = te

    train_x = tr_x
    train_y = tr_y
    valid_x = va_x
    valid_y = va_y
    test_x = te_x
    test_y = te_y

    return train_x, train_y, valid_x, valid_y, test_x, test_y


if __name__ == "__main__":
    path = r'/data/lisa/data/mnist/mnist.pkl.gz'
    train_x, train_y, _, _, _, _ = load_mnist(path)

    W = theano.shared(np.zeros(28 * 28, 10).astype(np.float32))
    b = theano.shared(np.zeros(10).astype(np.float32))
    params = [W, b]
    X = T.matrix('X')
    y = T.ivector('y')
    y_hat = T.nnet.softmax(T.dot(X, W) + b)
    cost = T.mean(T.nnet.categorical_crossentropy(y_hat, y))
    gradients = T.grad(cost, wrt=params)
    updates = lasagne.updates.adam(gradients, params, lr)
