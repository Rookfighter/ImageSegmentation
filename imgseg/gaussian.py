'''
Created on 15 Nov 2016

@author: Fabian Meyer
'''

import numpy as np
import scipy.stats

_DIM = 3
_EM_ITER = 10
_CONV_THRESHOLD = 0.5

def _expectation_maximization(img, gparam):
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

    likelihood = 0.0
    for em_it in range(_EM_ITER):
        print('EM iteration {}'.format(em_it))

        # E-Step: calculate posterior probability
        print('-- E-Step: Calc posterior probability ...')
        for i in range(gparam['k']):
            # likelihood function for data with current 'u' and 'cov'
            # calc probability for each sample with current estimated params
            postprob[i, :] = gparam['w'][i] * scipy.stats.multivariate_normal.pdf(img_vec,
                                                                        mean=gparam['u'][i],
                                                                        cov=gparam['cov'][i])
        # accum postprob of all gauss per sample
        postprob_sum = postprob.sum(0)
        # avoid division by 0
        postprob_sum[postprob_sum == 0] = 1
        # works the same as 'postprob[:, :] /= postprob_sum'
        # is elementwise division
        postprob /= postprob_sum
        # accum postprob of all samples per gauss
        postprob_sum = postprob.sum(1)

        # M-Step: reassign gauss parameters
        print('-- M-Step: Calc new gauss parameters ...')
        for i in range(gparam['k']):
            gparam['u'][i] = 0.0
            for n in range(img_vec.shape[0]):
                gparam['u'][i] += postprob[i, n] * img_vec[n]
            gparam['u'][i, :] /= postprob_sum[i]

            diff = np.subtract(img_vec, gparam['u'][i])
            gparam['cov'][i] = 0.0
            for n in range(img_vec.shape[0]):
                gparam['cov'][i] += postprob[i, n] * np.outer(diff[n], diff[n])
            gparam['cov'][i, :] /= postprob_sum[i]

            gparam['w'][i] = postprob_sum[i] / postprob.shape[0]

        # calculate current likelihood to check
        # if EM converges
#         print('-- Calc current likelihood ...')
#         likelihood_old = likelihood
#         for n in range(img_vec.shape[0]):
#             tmp = 0
#             for i in range(gparam['k']):
#                 tmp += gparam['w'][i] * scipy.stats.multivariate_normal.pdf(img_vec[n], mean=gparam['u'][i], cov=gparam['cov'][i])
#             likelihood += np.log(tmp)
#
#         print('-- Check convergence: ({}, {}) ...'.format(likelihood_old, likelihood))
#         if np.abs(likelihood - likelihood_old) <= _CONV_THRESHOLD:
#             print('Stop EM. Reached likelihood {}.'.format(likelihood))
#             break

def _get_box(x1, y1, x2, y2, image):
    return image[y1:y2, x1:x2]

def gaussian(img, k):
    gparamfg = {
        'u': np.empty((k, _DIM), dtype=np.float64),
        'cov': np.empty((k, _DIM, _DIM), dtype=np.float64),
        'w': np.empty(k, dtype=np.float64),
        'k': k
    }
    gparambg = {
        'u': np.empty((k, _DIM), dtype=np.float64),
        'cov': np.empty((k, _DIM, _DIM), dtype=np.float64),
        'w': np.empty(k, dtype=np.float64),
        'k': k
    }

    imgfg = _get_box(200, 285, 400, 330, img)
    imgbg = _get_box(50, 50, 200, 100, img)

    print('Foreground EM')
    _expectation_maximization(imgfg, gparamfg)
    print('Background EM')
    _expectation_maximization(imgbg, gparambg)

    print('Generate final gaussians')
    print('Foreground parameters:')
    print('-- u = {}'.format(gparamfg['u']))
    print('-- cov = {}'.format(gparamfg['cov']))
    print('-- w = {}'.format(gparamfg['w']))
    print('Background parameters:')
    print('-- u = {}'.format(gparambg['u']))
    print('-- cov = {}'.format(gparambg['cov']))
    print('-- w = {}'.format(gparambg['w']))
    # create gaussians with estimated params
    gaussfg = [scipy.stats.multivariate_normal(gparamfg['u'][i], gparamfg['cov'][i]) for i in range(k)]
    gaussbg = [scipy.stats.multivariate_normal(gparambg['u'][i], gparambg['cov'][i]) for i in range(k)]

    print('Create result picture ...')
    # create result picture on trained gaussians
    result = np.zeros((img.shape[0], img.shape[1], _DIM), dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            probfg = 0
            probbg = 0
            for i in range(k):
                probfg += gparamfg['w'][i] * gaussfg[i].pdf(img[y, x])
                probbg += gparambg['w'][i] * gaussbg[i].pdf(img[y, x])

            # if foreground has higher prob color the pixel yellow
            if probfg >= probbg:
                result[y, x, :] = np.array([255, 255, 0])
            else:
                result[y, x, :] = np.array([0, 0, 0])

    return result
