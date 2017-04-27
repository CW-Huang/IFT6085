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



clip_grad = 1
max_norm = 5

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

    def __init__(self, use_alpha=False):

        self.inpv = T.matrix('inpv')
        self.ep = T.tensor4('ep')
        self.w = T.scalar('w')
        self.lr = T.scalar('lr')      # learning rate

        batch_size = T.cast(self.inpv.shape[0], 'float32')

        self.loss_curr, self.params_curr = build_graph(self.inpv, self.ep, self.w)
        
        self.grads_curr = T.grad(self.loss_curr, self.params_curr)

        self.loss_prev, self.params_prev = build_graph(self.inpv, self.ep, self.w)
        self.grads_prev = T.grad(self.loss_prev, self.params_prev)

        # clipping
        self.grads_curr = lasagne.updates.total_norm_constraint(self.grads_curr,
                                                            max_norm=max_norm)
        self.grads_curr = [T.clip(g, -clip_grad, clip_grad) for g in self.grads_curr]

        self.grads_prev = lasagne.updates.total_norm_constraint(self.grads_prev,
                                                            max_norm=max_norm)
        self.grads_prev = [T.clip(g, -clip_grad, clip_grad) for g in self.grads_prev]

        self.grads_accumulate = [
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



        self.monitor_all_grads_covariance()
        self.monitor_grads_prev_variance()

        if not use_alpha:
            print "Normal SVRG"
            self.deltas = [
                g_c - g_p + (a / self.counter)
                for g_c, g_p, a in zip(self.grads_curr,
                                       self.grads_prev,
                                       self.grads_accumulate)
            ]
        else:
            print "SVRG-RE"
            self.deltas = [
               g_c - (mini_cov/(all_var + 1e-6)).clip(-10, 10)*(g_p - (a / self.counter))
               for g_c, g_p, a, mini_cov, all_var in zip(self.grads_curr,
                                      self.grads_prev,
                                      self.grads_accumulate,
                                      self.all_grads_covariance[2], # the covariance between both
                                      self.grads_prev_variance)]




        self.updates = lasagne.updates.sgd(self.deltas,
                                           self.params_curr,
                                           learning_rate=self.lr)

        self.monitor_updates_variance()

        self.reset_and_copy = theano.function(
            inputs=[], updates=(copy_to_prev_update +
                                reset_grads_update +
                                reset_count_update)
        )

        self.accumulate_gradients_func = theano.function(
            inputs=[self.inpv, self.ep, self.w],
            updates=acc_grads_update + count_update
        )



        
        print '\tgetting train func'
        self.train_func = theano.function(
            inputs=[self.inpv, self.ep, self.w, self.lr],
            outputs=self.loss_curr,
            updates=self.updates
        )



    def train(self,input,n_mc,n_iw,w,lr=lr_default):
        n = input.shape[0]
        ep = np.random.randn(n,n_mc,n_iw,ds[0]).astype(floatX)
        #print lr
        return self.train_func(input, ep, w, lr)

    def accumulate_gradients(self, input, n_mc, n_iw, w):
        n = input.shape[0]
        ep = np.random.randn(n, n_mc, n_iw, ds[0]).astype(floatX)
        return self.accumulate_gradients_func(input, ep, w)


    def accumulate_delta_var(self, input, n_mc, n_iw, w, lr):
        n = input.shape[0]
        ep = np.random.randn(n, n_mc, n_iw, ds[0]).astype(floatX)
        return self.accumulate_all_delta_func(input, ep, w, lr)

    def accumulate_grad_prev_var(self, input, n_mc, n_iw, w):
        n = input.shape[0]
        ep = np.random.randn(n, n_mc, n_iw, ds[0]).astype(floatX)
        return self.accumulate_all_grad_prev_func(input, ep, w)

    def monitor_updates_variance(self):
        shapes = [p.get_value().shape for p in self.params_curr]


        #Getting our update direction
        vars = [y -p for (x, y), p in zip(self.updates.items(), self.params_curr)]#[self.lr*g for g in self.deltas]
        all_cumulate, counter, accumulate_all_ms_func, reset_all, get_variable_variance_func, all_cov = self.tract_covariance(vars, None,
                                                                                                                                   shapes, 1,
                                                                                                                                   [
                                                                                                                                       self.inpv,
                                                                                                                                       self.ep,
                                                                                                                                       self.w,
                                                                                                                                   self.lr])

        self.all_cumulate = all_cumulate
        self.counter_delta = counter
        self.accumulate_all_delta_func = accumulate_all_ms_func
        self.reset_delta = reset_all
        self.get_delta_variance_func = get_variable_variance_func[0]

    def monitor_grads_prev_variance(self):
        shapes = [p.get_value().shape for p in self.params_curr]
        all_cumulate, counter, accumulate_all_ms_func, reset_all, get_variable_variance_func, all_cov = self.tract_covariance(self.grads_prev, None,
                                                                                                                                   shapes, 1,
                                                                                                                                   [
                                                                                                                                       self.inpv,
                                                                                                                                       self.ep,
                                                                                                                                       self.w])

        self.accumulate_grad_prev = all_cumulate
        self.counter_grad_prev = counter
        self.accumulate_all_grad_prev_func = accumulate_all_ms_func
        self.reset_grad_prev = reset_all
        self.get_grad_prev_variance_func = get_variable_variance_func[0]
        self.grads_prev_variance = all_cov[0]

    def monitor_all_grads_covariance(self):
        shapes = [p.get_value().shape for p in self.params_curr]
        all_cumulate, counter, accumulate_all_ms_func, reset_all, get_variable_variance_func, all_cov = self.tract_covariance(self.grads_curr, self.grads_prev,
                                                                                                                                   shapes, 1,
                                                                                                                                   [
                                                                                                                                       self.inpv,
                                                                                                                                       self.ep,
                                                                                                                                       self.w])

        self.accumulate_all_grad = all_cumulate
        self.counter_all_grad = counter
        self.accumulate_all_grad_func = accumulate_all_ms_func
        self.reset_all_grad = reset_all
        self.get_all_grad_variance_func = get_variable_variance_func
        self.all_grads_covariance = all_cov

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

        # acc_m1_update = {c: c for c in accumulate_m1}
        # for ex in [variable]:
        #     for cummul, grad in zip(accumulate_m1, ex):
        #         acc_m1_update[cummul] = acc_m1_update[cummul] + grad
        #
        # acc_m2_update = {c: c for c in accumulate_m2}
        # for ex in [variable]:
        #     for cummul, grad in zip(accumulate_m2, ex):
        #         acc_m1_update[cummul] = acc_m1_update[cummul] + grad
        #
        # acc_m11_update = {c: c for c in accumulate_m11}
        # for ex in [variable]:
        #     for cummul, grad in zip(accumulate_m11, ex):
        #         acc_m11_update[cummul] = acc_m11_update[cummul] + grad * grad
        #
        # acc_m22_update = {c: c for c in accumulate_m2}
        # for ex in [variable]:
        #     for cummul, grad in zip(accumulate_m2, ex):
        #         acc_m2_update[cummul] = acc_m2_update[cummul] + grad ** 2
        #
        #
        # acc_m12_update = {c: c for c in accumulate_m1}
        # for ex in [variable]:
        #     for cummul, grad in zip(accumulate_m1, ex):
        #         acc_m1_update[cummul] = acc_m1_update[cummul] + grad
        #
        # reset_all_ms_update = [
        #     (a, a * np.float32(0.))
        #     for a in accumulate_m1 + accumulate_m2
        #     ]
        #
        # variable_variance = [
        #     m2/counter - T.power(m1/counter, 2) for m1, m2 in zip(accumulate_m1, accumulate_m2)
        # ]
        #
        # acc_m2_update.update(acc_m1_update)
        # acc_m2_update.update(count_update)
        # accumulate_all_ms_func = theano.function(
        #     inputs=inputs,
        #     updates=acc_m2_update
        # )
        #
        # reset_all = theano.function(inputs=[], updates=reset_all_ms_update + reset_count_update)
        #
        # get_variable_variance_func = theano.function(
        #     inputs=[], outputs=variable_variance
        # )
        #
        # return accumulate_m1, accumulate_m2, counter, accumulate_all_ms_func, reset_all, get_variable_variance_func, variable_variance




    def accumulate_batch_cov(self, input, n_mc, n_iw, w):
        n = input.shape[0]
        ep = np.random.randn(n, n_mc, n_iw, ds[0]).astype(floatX)
        #return self.accumulate_cov_func(input, ep, w)
        return self.accumulate_all_grad_func(input, ep, w)


def train_model(model,epochs=10,bs=64,n_mc=1,n_iw=1, cumulate_cov=False, w=lambda t:1.):

    print '\n\ntraining with epochs:{}, batchsize:{}'.format(epochs,bs)

    def data_generator(batch_size=bs):
        for i in range(50000/bs):
            yield train_x[i*batch_size:(i+1)*batch_size].reshape(batch_size,28*28)

    t = 0
    lr =0.01
    records = list()
    updates_var = []


    for e in range(epochs):
        model.reset_and_copy()

        for x in data_generator():
            model.accumulate_gradients(x, n_mc, n_iw, w(t))

        model.reset_delta()
        for x in data_generator(1):

            # variance of the updates
            model.accumulate_delta_var(x, n_mc, n_iw, w(t), lr=lr)
            # variance of the grads.
            model.accumulate_grad_prev_var(x, n_mc, n_iw, w(t))


        print "The average update norm is:"
        vars = model.get_delta_variance_func()
        vars_norm = np.linalg.norm(vars[-2])
        print np.mean(vars_norm)

        print "The average variance norm is:"
        #vars = model.get_data_var(train_x.shape[0])
        vars = model.get_grad_prev_variance_func()
        vars_norm = np.linalg.norm(vars[-2])
        #print np.max(vars[-2])
        updates_var.append(vars_norm)

            
        i = 0
        for x in data_generator():

            #model.reset_cov()
            if cumulate_cov:
                model.reset_all_grad()
                #Get the alpha
                for ex in x:
                    model.accumulate_batch_cov(ex.reshape(1, ex.shape[0]), n_mc, n_iw, w(t))

                #covs = model.get_cov_func()
                covs = model.get_all_grad_variance_func[2]()
                #vars = model.get_all_grad_variance_func[1]()
                #vars = model.get_grad_prev_variance_func()
                alphas = [cov/var for cov, var in zip(covs, vars)]
                print np.mean(np.absolute(alphas[-2]))

            loss = model.train(x, n_mc, n_iw, w(t), lr=lr)
            records.append(loss)
            if t % 50 == 0:
                print "t: {}, e: {}, i: {}, loss: {}".format(t, e, i, loss)

            t+=1

            if t==10:
                break
            i += 1

    return records, updates_var

if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    path = r'/data/lisa/data/mnist/mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(path)

    tests = [
        [10,20,1,1, False],
        #[10*50,50*20,1,1, False],
        #[10,20,50,1,False],
        #[10,20,1,50, False]
    ]
    toplots = list()
    all_updates_var = []

    for test in tests:
        print '\n\nn_epochs:{}, batchsize:{}, n_mc:{}, n_iw:{}'.format(*test)
        model = VAE(use_alpha=test[-1])
        records, updates_var = train_model(model,*test)
        toplots.append(records)
        all_updates_var.append(updates_var)

    fig = plt.figure(figsize=(8,8))
    for i in range(len(tests)):
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(toplots[i])
        plt.title(tests[i])
        plt.ylim((80,250))
    plt.savefig('vae_iwae_example.jpg',format='jpeg')
    plt.savefig('vae_iwae_example.tiff',format='tiff')

    fig = plt.figure(figsize=(8,8))
    for i in range(len(tests)):
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(all_updates_var[i])
        plt.title(tests[i])
        #plt.ylim((80,250))
    plt.savefig('vae_svrg_var.jpg',format='jpeg')
    plt.savefig('vae_svrg_var.tiff',format='tiff')


