# -*- coding: utf-8 -*-
import cPickle as pickle
from sklearn.preprocessing import OneHotEncoder
import lasagne
from lasagne.layers import get_output
from lasagne.layers import get_all_params
from lasagne.objectives import binary_crossentropy as bc
import theano
import theano.tensor as T
import numpy as np

from utils import log_mean_exp, log_stdnormal , log_normal
from vae import *


def build_graph(inpv, ep, w):
    n, n_mc, n_iw, _ = ep.shape
    enc_m = get_encoder()
    enc_s = get_encoder()
    dec = get_decoder()
    mu = get_output(enc_m, inpv).dimshuffle(0,'x','x',1)
    log_s = get_output(enc_s, inpv).dimshuffle(0,'x','x',1)
    log_v = 2 * log_s
    sigma = T.exp(log_s)
    var = T.exp(log_s * 2)
    z = mu + sigma * ep
    z_reshaped = z.reshape((n * n_mc * n_iw, ds[0]))
    rec_reshaped = get_output(dec, z_reshaped)
    rec = rec_reshaped.reshape((n, n_mc, n_iw, ds[-1]))

    # lazy modeling just using binary crossentropy (non-binarized) ...
    log_px_z = - bc(rec, inpv.dimshuffle(0,'x','x',1))

    # KL divergence
    log_pz   = log_stdnormal(z)
    log_qz_x = log_normal(z,mu,log_v)
    kls_d = log_qz_x - log_pz

    losses_iw = - log_px_z.sum(3) + w * kls_d.sum(3)
    losses_mc = - log_mean_exp(-losses_iw,axis=2)
    
        
    loss = T.mean(losses_mc)

    params = (get_all_params(enc_m) +
              get_all_params(enc_s) +
              get_all_params(dec))

    return loss, params


#def compute_multiple_grad(losses, params):
#    
#    
#    def foo(i, W1, b2, W3, b4, W5, b6, W7, b8, W9, 
#            b10, W11, b12, W13, b14, W15, b16, W17, b18,
#            pW1, pb2, pW3, pb4, pW5, pb6, pW7, pb8, pW9,
#            pb10, pW11, pb12, pW13, pb14, pW15, pb16, pW17, pb18,
#           losses):
#        
#        params = [pW1, pb2, pW3, pb4, pW5, pb6, pW7, pb8, pW9,
#            pb10, pW11, pb12, pW13, pb14, pW15, pb16, pW17, pb18,]
#        
#        
#        grads = T.grad(losses[i].mean(), params)
#        return grads
#    
#    grads = theano.scan(fn=foo, sequences=theano.tensor.arange(losses.shape[0]),
#                              outputs_info=[T.zeros_like(p) for p in params],
#                              non_sequences=params + [losses], 
#                       n_steps=losses.shape[0])
#    
#    return grads

def get_batch_gradient(grads):


    # For now
    return None
    # batch_size = T.cast(inpv.shape[0], 'float32')
    #
    # instance_grads = []
    #
    # for g in grads:
    #
    #     instance_g = None
    #     print g
    #     print g.owner
    #     print g.owner.inputs
    #     print g.owner.inputs[0]
    #     print g.owner.inputs[0].owner
    #     if len(g.owner.inputs) == 2:
    #         layer_input = g.owner.inputs[0].dimshuffle(1, 0, 'x')
    #         delta_output = g.owner.inputs[1].dimshuffle(0, 'x', 1)
    #         instance_g = batch_size * layer_input * delta_output
    #     elif len(g.owner.inputs) == 1:
    #         instance_g = batch_size * (
    #             g.owner.inputs[0].owner.inputs[0].owner.inputs[0].owner.inputs[0].owner)
    #     else:
    #         print g.owner
    #
    #     instance_grads.append(instance_g)
    #
    # return instance_grads


class VAE(object):

    def __init__(self):

        self.inpv = T.matrix('inpv')
        self.ep = T.tensor4('ep')
        self.w = T.scalar('w')
        self.lr = T.scalar('lr')      # learning rate

        batch_size = T.cast(self.inpv.shape[0], 'float32')

        self.loss_curr, self.params_curr = build_graph(self.inpv, self.ep, self.w)
        
        self.grads_curr = T.grad(self.loss_curr, self.params_curr)

        self.loss_prev, self.params_prev = build_graph(self.inpv, self.ep, self.w)
        self.grads_prev = T.grad(self.loss_prev, self.params_prev)

        self.grads_accumulate = [
            theano.shared(np.zeros(p.get_value().shape).astype(np.float32))
            for p in self.params_curr
        ]

        # for the variance
        self.grads2_accumulate = [
            theano.shared(np.zeros(p.get_value().shape).astype(np.float32))
            for p in self.params_curr
            ]

        self.counter = theano.shared(np.float32(0.))
        copy_to_prev_update = zip(self.params_prev, self.params_curr)

        acc_grads_update = [
            (a, a + batch_size * g)
            for a, g in zip(self.grads_accumulate, self.grads_prev)
        ]
        count_update = [(
            self.counter, self.counter + batch_size
        )]

        reset_grads_update = [
            (a, a * np.float32(0.))
            for a, g in zip(self.grads_accumulate, self.grads_prev)
        ]
        reset_count_update = [(
            self.counter, np.float32(0.)
        )]

        # var ** 2
        # We assume that the minibatch size is one.
        acc_grads2_update = {c: c for c in self.grads2_accumulate}
        for ex in [self.grads_curr]:
            for cummul, grad in zip(self.grads2_accumulate, ex):
                acc_grads2_update[cummul] = acc_grads2_update[cummul] + grad ** 2

        reset_grads2_update = [
            (a, a * np.float32(0.))
            for a in self.grads2_accumulate
            ]

        self.accumulate_grads2_func = theano.function(
            inputs=[self.inpv, self.ep, self.w],
            updates=acc_grads_update
        )

        self.deltas = [
            g_c - g_p + (a / self.counter)
            for g_c, g_p, a in zip(self.grads_curr,
                                   self.grads_prev,
                                   self.grads_accumulate)
        ]

        self.updates = lasagne.updates.sgd(self.deltas,
                                           self.params_curr,
                                           learning_rate=self.lr)


        self.reset_and_copy = theano.function(
            inputs=[], updates=(copy_to_prev_update +
                                reset_grads_update +
                                reset_count_update + reset_grads2_update)
        )

        self.accumulate_gradients_func = theano.function(
            inputs=[self.inpv, self.ep, self.w],
            updates=acc_grads_update + count_update
        )
        
        self.get_grads = theano.function(
            [self.inpv, self.ep, self.w],
            self.grads_curr)
        
        print '\tgetting train func'
        self.train_func = theano.function(
            inputs=[self.inpv, self.ep, self.w, self.lr],
            outputs=self.loss_curr,
            updates=self.updates
        )

    def train(self,input,n_mc,n_iw,w,lr=lr_default):
        n = input.shape[0]
        ep = np.random.randn(n,n_mc,n_iw,ds[0]).astype(floatX)
        return self.train_func(input, ep, w, lr)

    def accumulate_gradients(self, input, n_mc, n_iw, w):
        n = input.shape[0]
        ep = np.random.randn(n, n_mc, n_iw, ds[0]).astype(floatX)
        return self.accumulate_gradients_func(input, ep, w)

    def accumulate_grads2(self, input, n_mc, n_iw, w):
        n = input.shape[0]
        ep = np.random.randn(n, n_mc, n_iw, ds[0]).astype(floatX)
        return self.accumulate_grads2_func(input, ep, w)
                   
    def get_all_grads(self, input, n_mc, n_iw, w):
        n = input.shape[0]
        ep = np.random.randn(n, n_mc, n_iw, ds[0]).astype(floatX)
                   
        grads = []
        for i, ex in enumerate(input):
            grad =  self.get_grads(ex.reshape(1, ex.shape[0]), 
                                   ep[i].reshape((1,)+ ep.shape[1:]), w)
            grads.append(grad)
                   
        return grads

    def get_var_batch(self, input, n_mc, n_iw, w):

        n = input.shape[0]
        ep = np.random.randn(n, n_mc, n_iw, ds[0]).astype(floatX)

        return self.batch_grads(input, ep, w)

    def get_data_var(self, n):

        params_accumulate_1 = [p.get_value() for p in self.grads_accumulate]
        params_accumulate_2 = [p.get_value() for p in self.grads2_accumulate]


        vars = []
        for m1, m2 in zip(params_accumulate_1, params_accumulate_2):

            p_var = m2 - np.power(m1, 2)
            vars.append(p_var/n)


        return vars


def train_model(model,epochs=10,bs=64,n_mc=1,n_iw=1,w=lambda t:1.):

    print '\n\ntraining with epochs:{}, batchsize:{}'.format(epochs,bs)

    def data_generator(batch_size=bs):
        for i in range(50000/bs):
            yield train_x[i*batch_size:(i+1)*batch_size].reshape(batch_size,28*28)

    t = 0
    records = list()

    for e in range(epochs):
        model.reset_and_copy()

        for x in data_generator():
            model.accumulate_gradients(x, n_mc, n_iw, w(t))

        for x in data_generator(1):
            model.accumulate_grads2(x, n_mc, n_iw, w(t))

        print "The average variance norm is:"
        vars = model.get_data_var(train_x.shape[0])
        vars_norm = [np.linalg.norm(g) for g in vars]
        print np.mean(vars_norm)

            
        i = 0
        for x in data_generator():
            loss = model.train(x, n_mc, n_iw, w(t))
            records.append(loss)
            if t % 50 == 0:
                print "t: {}, e: {}, i: {}, loss: {}".format(t, e, i, loss)

            t+=1

            if t==10:
                break
            i += 1

    return records

if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    path = r'/data/lisa/data/mnist/mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(path)

    tests = [
        [10,20,1,1],
        #[10*50,50*20,1,1],
        [10,20,50,1],
        [10,20,1,50]
    ]
    toplots = list()
    for test in tests:
        print '\n\nn_epochs:{}, batchsize:{}, n_mc:{}, n_iw:{}'.format(*test)
        model = VAE()
        records = train_model(model,*test)
        toplots.append(records)

    fig = plt.figure(figsize=(8,8))
    for i in range(len(tests)):
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(toplots[i])
        plt.title(tests[i])
        plt.ylim((80,250))
    plt.savefig('vae_iwae_example.jpg',format='jpeg')
    plt.savefig('vae_iwae_example.tiff',format='tiff')


