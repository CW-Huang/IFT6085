import gzip
import cPickle as pickle
import theano
import theano.tensor as T
import numpy as np
import lasagne
import math
import random

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

update_fun = {
    "adam": lasagne.updates.adam,
    "sgd": lasagne.updates.sgd,
    "momentum": lasagne.updates.momentum,
}

if __name__ == "__main__":
    path = r'/data/lisa/data/mnist/mnist.pkl.gz'
    batch_size = 64
    epochs = 20
    lr = 0.001

    train_x, train_y, _, _, _, _ = load_mnist(path)
    data_x = theano.shared(train_x.astype(np.float32))
    data_y = theano.shared(train_y.astype(np.int32))

    W = theano.shared(np.zeros((28 * 28, 10)).astype(np.float32))
    b = theano.shared(np.zeros((10,)).astype(np.float32))
    params = [W, b]

    X = T.matrix('X')
    y = T.ivector('y')
    y_hat = T.nnet.softmax(T.dot(X, W) + b)
    loss = T.mean(T.nnet.categorical_crossentropy(y_hat, y))
    cost = loss + 1e-4 * sum(T.sum(T.sqr(w)) for w in params)
    gradients = T.grad(cost, wrt=params)

    for update_type in ["sgd"]:
        updates = update_fun[update_type](gradients, params, lr)
        deltas = [updates[p] - p for p in params]

        idx = T.iscalar('idx')
        train = theano.function(
            inputs=[idx],
            outputs=loss,
            updates=updates,
            givens={
                X: data_x[idx * batch_size:(idx + 1) * batch_size],
                y: data_y[idx * batch_size:(idx + 1) * batch_size],
            }
        )

        per_instance_gradient = theano.function(
            inputs=[idx],
            outputs=gradients,
            givens={
                X: data_x[idx:(idx + 1)],
                y: data_y[idx:(idx + 1)],
            }
        )

        def calculate_variance():
            acc_grad = None
            acc_grad_sqr = None
            count = train_x.shape[0]
            for i in xrange(count):
                grads = per_instance_gradient(i)
                grads = [np.array(g) for g in grads]
                if acc_grad is None:
                    acc_grad = [np.zeros(g.shape).astype(np.float32)
                                for g in grads]
                    acc_grad_sqr = [np.zeros(g.shape).astype(np.float32)
                                    for g in grads]
                for ag, ags, g in zip(acc_grad, acc_grad_sqr, grads):
                    ag += g
                    ags += g**2
            variances = [ags / count - (ag / count)**2
                         for ag, ags in zip(acc_grad, acc_grad_sqr)]
            return variances

        batches = int(math.ceil(train_x.shape[0] / float(batch_size)))
        iterations = 0

        with gzip.open("%s_method_log.pkl.gz" % update_type, 'wb') as f:
            for epoch in xrange(epochs):
                batch_idxs = range(batches)
                random.shuffle(batch_idxs)
                for i in batch_idxs:
                    logitem = {"cost": train(i)}
                    print logitem["cost"]
                    if iterations % 50 == 0:
                        variances = calculate_variance()
                        logitem["grad_variance"] = variances
                    iterations += 1
                    pickle.dump(logitem, f, 2)
