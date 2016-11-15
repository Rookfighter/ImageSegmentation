'''
Created on 15 Nov 2016

@author: Fabian Meyer
'''

import numpy as np
import scipy.stats

_DIM = 3
_EM_ITER = 1

def expectation_maximization(img, gparam):
    # set initial paramaters
    for i in range(gparam['k']):
        y, x = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])

        # 'u' as random pixel
        # 'cov' as identity * fac
        # 'w' as uniform weight
        gparam['u'][i][:] = img[y, x]
        gparam['cov'][i] = 3 * np.identity(_DIM)
        gparam['w'][i] = 1.0 / gparam['k']

    # reshape image matrix as a vector
    img_vec = img.reshape((img.shape[0] * img.shape[1], _DIM))
    # initialize vector for posterior prob
    postprob = np.empty((gparam['k'], img_vec.shape[0]))

    for _ in range(_EM_ITER):

        # E-Step: calculate posterior probability
        for i in range(gparam['k']):
            postprob[i, :] = gparam['w'][i] * scipy.stats.multivariate_normal.pdf(img_vec,
                                                                        mean=gparam['u'][i],
                                                                        cov=gparam['cov'][i])
        postprob_sum = postprob.sum(0)
        print(postprob)
        # avoid division by 0
        postprob_sum[postprob_sum == 0] = 1
        postprob /= postprob_sum
        postprob_sum = postprob.sum(1)
        print(postprob)
        # M-Step: reassign gauss parameters

        for i in range(gparam['k']):
            gparam['u'][i] = 0.0
            for n in range(img_vec.shape[0]):
                gparam['u'][i] += postprob[i, n] * img_vec[n]
            gparam['u'][i][:] /= postprob_sum[i]

            diff = np.subtract(img_vec, gparam['u'][i])
            gparam['cov'][i] = 0.0
            for n in range(img_vec.shape[0]):
                gparam['cov'][i] += postprob[i, n] * np.outer(diff[n], diff[n])
            gparam['cov'][i][:] /= postprob_sum[i]

            gparam['w'][i] = postprob_sum[i] / postprob.shape[0]

def gaussian(img, k):
    gparam = {
        'u': np.empty((k, _DIM), dtype=np.float64),
        'cov': np.empty((k, _DIM, _DIM), dtype=np.float64),
        'w': np.empty(k, dtype=np.float64),
        'k': k
    }

    expectation_maximization(img, gparam)
