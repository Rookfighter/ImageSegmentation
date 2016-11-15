'''
Created on 15 Nov 2016

@author: Fabian Meyer
'''

import scipy.misc
from imgseg import kmeans
from imgseg import gaussian

if __name__ == '__main__':

    print('Loading image "res/banana.png" ...')
    img = scipy.misc.imread('res/banana.png')
    print(img.shape)
    # print('Calculating kmeans ...')
    # result = kmeans.kmeans(img, 2)
    # print('Saving image "result.png" ...')
    # scipy.misc.imsave('result.png', result)

    print('Calculating gaussian ...')
    gaussian.gaussian(img, 2)

