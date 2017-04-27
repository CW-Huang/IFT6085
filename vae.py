# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:35:08 2017

@author: Chin-Wei


VAE with monte carlo sampling + importance sampling

"""


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

floatX = theano.config.floatX
nonl = lasagne.nonlinearities.tanh
linear = lasagne.nonlinearities.linear
sigmoid = lasagne.nonlinearities.sigmoid


ds = [50,200,200,784]
lr_default = 0.001 # learning rate
clip_grad = 1
max_norm = 5

def get_encoder():

    print '\tgetting encoder'
    enc = lasagne.layers.InputLayer(shape=(None,ds[-1]))

    for d in ds[1:-1][::-1]:
        enc = lasagne.layers.DenseLayer(enc,d,nonlinearity=nonl)
        print enc.output_shape

    enc = lasagne.layers.DenseLayer(enc,ds[0],nonlinearity=linear)
    print enc.output_shape

    return enc

def get_decoder():

    print '\tgetting decoder'
    dec = lasagne.layers.InputLayer(shape=(None,ds[0]))

    for d in ds[1:-1]:
        dec = lasagne.layers.DenseLayer(dec,d,nonlinearity=nonl)
        print dec.output_shape

    dec = lasagne.layers.DenseLayer(dec,ds[-1],nonlinearity=sigmoid)
    print dec.output_shape


    return dec



def load_mnist(path):

    import gzip
    tr,va,te = pickle.load(gzip.open(path,'r'))
#    tr,va,te = pickle.load(open(path,'r'))

    tr_x,tr_y = tr
    va_x,va_y = va
    te_x,te_y = te

    enc = OneHotEncoder(10)

    tr_y = enc.fit_transform(tr_y).toarray().reshape(50000,10).astype(int)
    va_y = enc.fit_transform(va_y).toarray().reshape(10000,10).astype(int)
    te_y = enc.fit_transform(te_y).toarray().reshape(10000,10).astype(int)

    train_x = tr_x
    train_y = tr_y
    valid_x = va_x
    valid_y = va_y
    test_x = te_x
    test_y = te_y

    return train_x, train_y, valid_x, valid_y, test_x, test_y



class VAE(object):

    def __init__(self):

        self.inpv = T.matrix('inpv')
        self.ep = T.tensor4('ep')
        self.w = T.scalar('w')
        self.n_iw = T.iscalar('n_iw') # number of importance samples
        self.n_mc = T.iscalar('n_mc') # number of monte carlo samples
        self.lr = T.scalar('lr')      # learning rate
        self.n = self.inpv.shape[0]

        self.enc_m = get_encoder()
        self.enc_s = get_encoder()
        self.dec = get_decoder()

        self.mu = get_output(self.enc_m,self.inpv).dimshuffle(0,'x','x',1)
        self.log_s = get_output(self.enc_s,self.inpv).dimshuffle(0,'x','x',1)
        self.log_v = 2*self.log_s
        self.sigma = T.exp(self.log_s)
        self.var = T.exp(self.log_s*2)
        self.z = self.mu + self.sigma * self.ep
        self.z_reshaped = self.z.reshape(
            (self.n*self.n_mc*self.n_iw,ds[0])
        )
        self.rec_reshaped = get_output(self.dec,self.z_reshaped)
        self.rec = self.rec_reshaped.reshape(
            (self.n,self.n_mc,self.n_iw,ds[-1])
        )

        # lazy modeling just using binary crossentropy (non-binarized) ...
        self.log_px_z = - bc(self.rec,
                             self.inpv.dimshuffle(0,'x','x',1))
        self.log_pz   = log_stdnormal(self.z)
        self.log_qz_x = log_normal(self.z,self.mu,self.log_v)

        self.kls_d = self.log_qz_x - self.log_pz

        self.losses_iw = - self.log_px_z.sum(3) + \
                           self.w * self.kls_d.sum(3)
        self.losses_mc = - log_mean_exp(-self.losses_iw,axis=2)
        self.loss = T.mean(self.losses_mc)


        self.params = get_all_params(self.enc_m) + \
                      get_all_params(self.enc_s) + \
                      get_all_params(self.dec)
        self.updates = lasagne.updates.adam(self.loss,self.params,self.lr)

        self.grads = T.grad(self.loss, self.params)

        self.mgrads = lasagne.updates.total_norm_constraint(self.grads,
                                                            max_norm=max_norm)
        self.cgrads = [T.clip(g, -clip_grad, clip_grad) for g in self.mgrads]

        self.updates = lasagne.updates.adam(self.cgrads,
                                            self.params,
                                            beta1=0.9,
                                            beta2=0.999,
                                            epsilon=1e-4,
                                            learning_rate=self.lr)

        print '\tgetting train func'
        self.train_func = theano.function([self.inpv,self.ep,self.w,
                                           self.n_mc,self.n_iw,self.lr],
                                           self.loss.mean(),
                                           updates=self.updates)

        #print '\tgetting other useful funcs'
        #self.recon = theano.function([self.inpv,self.ep,self.n_mc,self.n_iw],
        #                             self.rec)
        #self.project = theano.function([self.inpv,self.ep],self.z)
        #self.get_mu = theano.function([self.inpv],self.mu)
        #self.get_var = theano.function([self.inpv],self.var)

        self.monitor_updates_variance()


    def tract_covariance(self, variable1, variable2, shapes, inc, inputs):

        """
        Return the covariance of variable 1 and 2, and the variance of v1 and v2.
         If only variable 1 is present, only return var(v1).
        :param variable1:
        :param variable2:
        :param shapes:
        :param inc:
        :param inputs:
        :return:
        """

        # all of our moments
        accumulate_m1 = [
            theano.shared(np.zeros(p).astype(np.float32))
            for p in shapes
            ]

        accumulate_m11 = [
            theano.shared(np.zeros(p).astype(np.float32))
            for p in shapes
            ]

        accumulate_m12 = []
        accumulate_m22 = []
        accumulate_m2 = []

        if variable2 is not None:
            accumulate_m12 = [
            theano.shared(np.zeros(p).astype(np.float32))
            for p in shapes
            ]


            accumulate_m22 = [
                theano.shared(np.zeros(p).astype(np.float32))
                for p in shapes
                ]


            accumulate_m2 = [
                theano.shared(np.zeros(p).astype(np.float32))
                for p in shapes
                ]

        counter = theano.shared(np.float32(0.))
        count_update = {counter: counter + inc}
        reset_count_update = [(
            counter, np.float32(0.)
        )]

        # var ** 2
        # We assume that the minibatch size is one.
        # Our moment updates

        def first_moment(acc_m, v):

            if v is None:
                return {}

            acc_m_update = {c: c for c in acc_m}
            for ex in [v]:
                for cummul, grad in zip(acc_m, ex):
                    acc_m_update[cummul] = acc_m_update[cummul] + grad
            return acc_m_update

        def second_moment(acc_m, v1, v2):

            if v1 is None or v2 is None:
                return {}

            acc_m_update = {c: c for c in acc_m}
            for cummul, grad1, grad2 in zip(acc_m, v1, v2):
                acc_m_update[cummul] = acc_m_update[cummul] + grad1 * grad2
            return acc_m_update

        acc_m1_update = first_moment(accumulate_m1, variable1)
        acc_m2_update = first_moment(accumulate_m2, variable2)
        acc_m11_update = second_moment(accumulate_m11, variable1, variable1)
        acc_m12_update = second_moment(accumulate_m12, variable1, variable2)
        acc_m22_update = second_moment(accumulate_m22, variable2, variable2)

        cov12 = [m2/counter - m1_1*m1_2 for m1_1, m1_2, m2 in zip(accumulate_m1, accumulate_m2, accumulate_m12)]
        cov11 = [m2/counter - m1_1*m1_2 for m1_1, m1_2, m2 in zip(accumulate_m1, accumulate_m1, accumulate_m11)]
        cov22 = [m2/counter - m1_1*m1_2 for m1_1, m1_2, m2 in zip(accumulate_m2, accumulate_m2, accumulate_m22)]

        reset_all_ms_update = [
            (a, a * np.float32(0.))
            for a in accumulate_m1 + accumulate_m2 + accumulate_m11 + accumulate_m12 + accumulate_m22
            ]

        updates = {}
        updates.update(acc_m1_update)
        updates.update(acc_m2_update)
        updates.update(acc_m11_update)
        updates.update(acc_m12_update)
        updates.update(acc_m22_update)
        updates.update(count_update)


        accumulate_all_ms_func = theano.function(
            inputs=inputs,
            updates=updates
        )

        reset_all = theano.function(inputs=[], updates=reset_all_ms_update + reset_count_update)

        get_cov11_func = theano.function(
            inputs=[], outputs=cov11
        )

        get_cov22_func = None
        get_cov12_func = None

        if variable2 is not None:
            get_cov22_func = theano.function(
                inputs=[], outputs=cov22
            )

            get_cov12_func = theano.function(
                inputs=[], outputs=cov12
            )

        all_cumulate = [accumulate_m1, accumulate_m2, accumulate_m11, accumulate_m12, accumulate_m22]

        return all_cumulate, counter, accumulate_all_ms_func, reset_all, [get_cov11_func, get_cov22_func, get_cov12_func], [cov11, cov22, cov12]

    def monitor_updates_variance(self):
        shapes = [p.get_value().shape for p in self.params]


        #Getting our update direction
        vars = [self.updates[p] - p for p in self.params]

        all_cumulate, counter, accumulate_all_ms_func, reset_all, get_variable_variance_func, all_cov = self.tract_covariance(vars, None,
                                                                                                                                   shapes, 1,
                                                                                                                              [
                                                                                                                                  self.inpv,
                                                                                                                                  self.ep,
                                                                                                                                  self.w,
                                                                                                                                  self.n_mc,
                                                                                                                                  self.n_iw,
                                                                                                                                  self.lr])

        self.all_cumulate = all_cumulate
        self.counter_delta = counter
        self.accumulate_all_delta_func = accumulate_all_ms_func
        self.reset_delta = reset_all
        self.get_delta_variance_func = get_variable_variance_func[0]




    def train(self,input,n_mc,n_iw,w,lr=lr_default):
        n = input.shape[0]
        ep = np.random.randn(n,n_mc,n_iw,ds[0]).astype(floatX)
        return self.train_func(input,ep,w,n_mc,n_iw,lr)

    def accumulate_gradients(self, input, n_mc, n_iw, w):
        n = input.shape[0]
        ep = np.random.randn(n, n_mc, n_iw, ds[0]).astype(floatX)
        return self.accumulate_gradients_func(input, n_mc, n_iw, ep, w)

    def accumulate_delta_var(self, input, n_mc, n_iw, w, lr=lr_default):
        n = input.shape[0]
        ep = np.random.randn(n, n_mc, n_iw, ds[0]).astype(floatX)
        return self.accumulate_all_delta_func(input, ep, w, n_mc, n_iw, lr)

    def get_data_var(self, n):

        params_accumulate_1 = [p.get_value() for p in self.grads_accumulate]
        params_accumulate_2 = [p.get_value() for p in self.grads2_accumulate]


        vars = []
        for m1, m2 in zip(params_accumulate_1, params_accumulate_2):

            p_var = m2 - np.power(m1, 2)
            vars.append(p_var/n)


        return [vars[-2]]

def train_model(model,epochs=10,bs=64,n_mc=1,n_iw=1,w=lambda t:1.):

    print '\n\ntraining with epochs:{}, batchsize:{}'.format(epochs,bs)

    t = 0
    records = list()
    for e in range(epochs):


        # For the variance:

        #model.reset()
        #for i in range(50000/bs):
        #    x = train_x[i*bs:(i+1)*bs].reshape(bs,28*28)
        #    model.accumulate_gradients(x, n_mc, n_iw, w(t))

        model.reset_delta()
        for i in range(50000/bs):
            x = train_x[i:(i+1)].reshape(1,28*28)
            model.accumulate_delta_var(x,n_mc,n_iw,w(t))

        print "The average update norm is:"
        vars = model.get_delta_variance_func()
        vars_norm = np.linalg.norm(vars[-2])
        print np.mean(vars_norm)


        for i in range(50000/bs):
            x = train_x[i*bs:(i+1)*bs].reshape(bs,28*28)

            loss = model.train(x,n_mc,n_iw,w(t))
            records.append(loss)

            if t%50 == 0:
                print t,e,i, loss
            t+=1

            if t==10:
                break

    return records


def train_model2(model,bs=20,n_mc=1,n_iw=1,w=lambda t:1.):

    print '\n\ntraining with bs:{}, n_mc:{}, n_iw:{}'.format(bs,n_mc,n_iw)

    t = 0
    records = list()
    for j in range(7):
        epochs = 3**j
        lr = 0.001 * 10**(-j/7.)
        for e in range(epochs):
            for i in range(50000/bs):
                x = train_x[i*bs:(i+1)*bs].reshape(bs,28*28)
                loss = model.train(x,n_mc,n_iw,w(t),lr)

                if t%50 == 0:
                    print t,e,i, loss
                t+=1
            records.append(loss)

    return records


if __name__ == '__main__':

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    path = r'/data/lisa/data/mnist/mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(path)


#==============================================================================
# decreasing learning rate (long)
#==============================================================================
#    tests = [
#        [20,1,1],
#        [50*20,1,1],
#        [20,50,1],
#        [20,1,50]
#    ]
#
#    toplots = list()
#
#    for test in tests:
#        print '\n\nbatchsize:{}, n_mc:{}, n_iw:{}'.format(*test)
#        model = VAE()
#        records = train_model(model,*test)
#        toplots.append(records)
#
#
#    fig = plt.figure(figsize=(8,8))
#    for i in range(len(tests)):
#        plt.plot(toplots[i])
#
#    plt.ylim((80,250))
#    plt.legend(map(lambda test:'bs:{}, n_mc:{}, n_iw:{}'.format(*test),tests),
#               loc=1)
#    plt.savefig('vae_iwae_example.jpg',format='jpeg')
#    plt.savefig('vae_iwae_example.tiff',format='tiff')


#==============================================================================
# fix learning rate
#==============================================================================

    tests = [
        [10,20,1,1],
        #[10*50,50*20,1,1],
        #[10,20,50,1],
        #[10,20,1,50]
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


#==============================================================================
# norm of variance of gradient estimate
#==============================================================================
#    # variance of gradient
#     model = VAE()
#
#     bs = 64
#     for i in range(50000/bs):
#        x = train_x[i*bs:(i+1)*bs].reshape(bs,28*28)
#        loss = model.train(x,1,1,1.)
#
#
#     f_grads = theano.function([model.inpv,model.ep,model.w,
#                               model.n_mc,model.n_iw],
#                              model.grads)
#     rms = lambda xx: (xx**2).mean()**0.5
#     l2norm = lambda xx: (xx**2).sum()**0.5
#
#     x = train_x[0].reshape(1,28*28)
#     n = x.shape[0]
#
#
#     norm_operator = rms
#     def get_norm_var(mc,iw):
#        norms = list()
#        for i in range(100):
#            ep = np.random.randn(n,mc,iw,ds[0]).astype(floatX)
#            norms.append(f_grads(x,ep,1.,mc,iw))
#        return [norm_operator(xx) for xx in np.array(norms).var(0)]
#
#     tests = [
#        [1,1],
#        [1,10],
#        [1,100],
#        [10,1],
#        [100,1]
#     ]
#
#     for (mc,iw) in tests:
#        plt.plot(get_norm_var(mc,iw))
#
#
#     plt.legend(['mc{}iw{}'.format(mc,iw) for (mc,iw) in tests],loc=1)
#     plt.savefig('rms_var_params.jpg',format='jpeg')

