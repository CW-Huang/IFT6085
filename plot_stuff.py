


import cPickle as pickle


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":

    #model = "svrg"
    #model = "adam"
    model = "sgd"
    #model = "svrgre"

    nll_f = open("{}_loss_10.pkl".format(model), 'r')
    var_f = open("{}_variance_10.pkl".format(model), 'r')
    nll_name = 'vae_{}_nll.jpg'.format(model)
    var_name = 'vae_{}_var.jpg'.format(model)

    nll_data = pickle.load(nll_f)
    var_data = pickle.load(var_f)


    print "plotting the nll"

    fig = plt.figure(figsize=(8,8))
    for i in range(len(nll_data)):
        ax = fig.add_subplot(1,1,i+1)
        ax.plot(nll_data[i])
        #plt.title(tests[i])
        #plt.ylim((80,250))
        plt.savefig(nll_name, format='jpeg')



    print "plotting the variance"
    var_data = np.array(var_data)
    fig = plt.figure(figsize=(8,8))
    for i in range(len(nll_data)):
        ax = fig.add_subplot(1,1,i+1)

        nb_p = var_data[i].shape[1]
        for p in range(nb_p)[-6:]:

            ax.semilogy(var_data[i, :, p])

        plt.legend(range(nb_p)[-6:])
        #plt.title(tests[i])
        #plt.ylim((80,250))
        plt.savefig(var_name, format='jpeg')

    #print np.array(var_data).shape
