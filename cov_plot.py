

import cPickle as pickle


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":


    cov_f = open("svrg_covnco_10_bak.pkl")
    cov_data = pickle.load(cov_f)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1)
    idx = []

    for i in range(18):

        if len(cov_data[0][i].shape) == 1:
            continue


        idx.append(i)

        cov_11 = cov_data[0][i].flatten()
        cov_22 = cov_data[1][i].flatten()
        cov_12 = cov_data[2][i].flatten()

        bins = range(21)
        bins = [b/10. - 1 for b in bins]
        # the histogram of the data with histtype='step'

        cor = cov_12/(np.sqrt(cov_11 + 1e-6)*np.sqrt(cov_22 + 1e-6))
        ratio =(np.sqrt(cov_11 + 1e-6)/np.sqrt(cov_22 + 1e-6))
        #print np.mean(cov_12)
        #print np.mean(cov_11)
        #print np.mean(cov_22)
        #print np.min(cor)
        n, bins, patches = ax.hist(cor, bins, histtype='bar', rwidth=0.8)



    plt.xlim(-.25, 1.0)
    legends = ["W{}".format(i+1) for i in range(9)]

    ax.legend(legends, loc=2)
    #plt.savefig('svrg_cor_bin.jpg', format='jpeg')
    #plt.close()

    ax = fig.add_subplot(1, 2, 2)

    for i in idx:
        cov_11 = cov_data[0][i].flatten()
        cov_22 = cov_data[1][i].flatten()
        cov_12 = cov_data[2][i].flatten()

        bins = range(21)
        bins = [b / 10. - 1 for b in bins]
        # the histogram of the data with histtype='step'

        cor = cov_12 / (np.sqrt(cov_11 + 1e-6) * np.sqrt(cov_22 + 1e-6))
        ratio = (np.sqrt(cov_11 + 1e-6) / np.sqrt(cov_22 + 1e-6))
        #print np.mean(cov_12)
        #print np.mean(cov_11)
        #print np.mean(cov_22)
        #print np.min(cor)


        n, bins, patches = ax.hist(ratio, bins, histtype='bar', rwidth=0.8)


    plt.xlim(.5, 1.0)
    ax.legend(legends, loc=2)
    plt.savefig('svrg_ratio_3_bin.jpg', format='jpeg')