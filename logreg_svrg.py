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


def build_model(X, y, W, b):
    params = [W, b]
    y_hat = T.nnet.softmax(T.dot(X, W) + b)
    loss = T.mean(T.nnet.categorical_crossentropy(y_hat, y))
    cost = loss + 1e-4 * sum(T.sum(T.sqr(w)) for w in params)
    gradients = T.grad(cost, wrt=params)
    return cost, params, gradients


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
    W_ = theano.shared(np.zeros((28 * 28, 10)).astype(np.float32))
    b_ = theano.shared(np.zeros((10,)).astype(np.float32))
    reg_W = theano.shared(np.zeros((28 * 28, 10)).astype(np.float32))
    reg_b = theano.shared(np.zeros((10,)).astype(np.float32))

    mean_prev = [reg_W, reg_b]

    X = T.matrix('X')
    y = T.ivector('y')
    cost, params, gradients = build_model(X, y, W, b)
    _, prev_params, prev_gradients = build_model(X, y, W_, b_)

    update_type = "sgd"

    updates = update_fun[update_type](gradients, params, lr)
    prev_updates = update_fun[update_type](prev_gradients, prev_params, lr)

    deltas = [-updates[p] + p for p in params]
    prev_deltas = [-prev_updates[p] + p for p in prev_params]

    calculate_reg = theano.function(
        inputs=[],
        updates=[(w, g) for w, g in zip(mean_prev, prev_deltas)],
        givens={X: data_x, y: data_y}
    )
    copy_params = theano.function(
        inputs=[],
        updates=[(_w, w) for _w, w in zip(prev_params, params)]
    )
    svrg_updates = [(p, p - (d - pd + reg))
                    for p, d, pd, reg in zip(params, deltas, prev_deltas,
                                             mean_prev)]
    idx = T.iscalar('idx')
    train = theano.function(
        inputs=[idx],
        outputs=cost,
        updates=svrg_updates,
        givens={
            X: data_x[idx * batch_size:(idx + 1) * batch_size],
            y: data_y[idx * batch_size:(idx + 1) * batch_size],
        }
    )

    per_instance_gradient = theano.function(
        inputs=[idx],
        outputs=deltas + prev_deltas,
#        outputs=[(d - pd + reg)
#                 for d, pd, reg in zip(deltas, prev_deltas, mean_prev)],
        givens={
            X: data_x[idx:(idx + 1)],
            y: data_y[idx:(idx + 1)],
        }
    )

    def calculate_variance():
        sum_grad = None
        sum_prev_grad = None
        sum_grad_sqr = None
        sum_grad_prev_grad = None

        count = train_x.shape[0]
        for i in xrange(count):
            grads = per_instance_gradient(i)
            grads = [np.array(g) for g in grads]
            curr_grads = grads[:len(grads) // 2]
            prev_grads = grads[len(grads) // 2:]

            if sum_grad is None:
                sum_grad = [np.zeros(g.shape).astype(np.float32) for g in grads]
                sum_prev_grad = [np.zeros(g.shape).astype(np.float32) for g in grads]
                sum_grad_grad = [np.zeros(g.shape).astype(np.float32) for g in grads]
                sum_grad_prev_grad = [np.zeros(g.shape).astype(np.float32) for g in grads]

            for sg, spg, sgg, sgpg, g, pg in zip(sum_grad, sum_prev_grad,
                                                 sum_grad_grad, sum_grad_prev_grad,
                                                 curr_grads, prev_grads):

                sg += g
                spg += pg
                sgg += g * g
                sgpg += pg * g

        grad_variance = [sgg / count - (sg / count)**2
                         for sg, sgg in zip(sum_grad, sum_grad_grad)]
        grad_covariance = [sgpg / count - (sg / count) * (spg / count)
                           for sg, spg, sgpg in zip(sum_grad, sum_prev_grad,
                                                    sum_grad_prev_grad)]
        return grad_variance, grad_covariance

    batches = int(math.ceil(train_x.shape[0] / float(batch_size)))
    iterations = 0

    with gzip.open("%s_svrg_method_log.pkl.gz" % update_type, 'wb') as f:
        for epoch in xrange(epochs):
            copy_params()
            calculate_reg()
            batch_idxs = range(batches)
            random.shuffle(batch_idxs)
            for i in batch_idxs:
                logitem = {"cost": train(i)}
                print logitem["cost"]
                if iterations % 50 == 0:
                    grad_variances, grad_covariances = calculate_variance()
                    logitem["grad_variances"] = grad_variances
                    logitem["grad_corvariances"] = grad_covariances_variances

                iterations += 1
                pickle.dump(logitem, f, 2)
