


import cPickle as pickle


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal



if __name__ == "__main__":

    #model = "svrg"
    #model = "svrg_2"
    #model = "adam"
    #model = "adam_all_4"
    #model = "sgd_all_4"
    #model = "svrg_20"
    #model = "momentum_all_4"
    #model = "svrgre_2"
    models = ["svrg_2", "svrgre_3", "sgd_all_4", "adam_all_4", "momentum_all_4"]
    titles = ["svrg", "resvrg", "sgd", "adam", "momentum"]
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 1, 1)

    for no_mo, (model, title) in enumerate(zip(models, titles)):


        nll_f = open("{}_loss_10.pkl".format(model), 'r')
        var_f = open("{}_variance_10.pkl".format(model), 'r')
        #nll_name = 'vae_{}_nll_2.jpg'.format(model)
        #var_name = 'vae_{}_var_2.jpg'.format(model)

        print "{}_loss_10.pkl".format(model), "{}_variance_10.pkl".format(model)

        nll_data = pickle.load(nll_f)
        var_data = pickle.load(var_f)

        if "momentum" in model:
            nll_data = [nll_data[1]]
            var_data = [var_data[1]]

        print "plotting the nll"
        #ax = fig.add_subplot(2, 5, no_mo+1)
        #plt.title(title + " nll")

        #fig = plt.figure(figsize=(8,8))
        for i in range(len(nll_data)):
            #ax = fig.add_subplot(1,1,i+1)
            nll = scipy.signal.medfilt(nll_data[i], 51)
            ax.plot(nll)
            plt.ylim((80,250))
            #plt.savefig(nll_name, format='jpeg')



        # print "plotting the variance"
        # ax = fig.add_subplot(2, 5, 5 + no_mo+1)
        # plt.title(title + " variance")
        #
        #
        # var_data = np.array(var_data)
        # #fig = plt.figure(figsize=(8,8))
        #
        # legend = ["W1", "b1", "W2", 'b2', 'W3', 'b3']
        #
        # for i in range(len(nll_data)):
        #     #ax = fig.add_subplot(1,1,i+1)
        #
        #     nb_p = var_data[i].shape[1]
        #     for p in range(nb_p)[-6:]:
        #
        #         ax.semilogy(var_data[i, :, p])
        #
        #         #if title == "svrg":
        #         #    print var_data[i, :, p]
        #
        #     plt.legend(legend, loc=3)
        #     #plt.title(tests[i])
        #     plt.ylim((1e-10, 1e-4))
        #     #plt.savefig(var_name, format='jpeg')
        #     #plt.close()


            #print np.array(var_data).shape

    plt.legend(titles)
    plt.savefig("var_n_nll_all.jpg", format='jpeg')
