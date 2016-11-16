'''
Created on 15 Nov 2016

@author: Fabian Meyer
'''

import scipy.misc
from imgseg import kmeans
from imgseg import gaussian

_SRC_FILE = 'res/banana.png'
_DEST_FILE = 'result.png'

def load_img():
    print('Loading image "{}" ...'.format(_SRC_FILE))
    return scipy.misc.imread(_SRC_FILE)

def save_result(result):
    print('Saving image "{}" ...'.format(_DEST_FILE))
    scipy.misc.imsave(_DEST_FILE, result)

if __name__ == '__main__':


    img = load_img()
    # print('Calculating kmeans ...')
    # result = kmeans.kmeans(img, 2)
    # print('Saving image "result.png" ...')
    # scipy.misc.imsave('result.png', result)

    print('Calculating gaussian ...')
    result = gaussian.gaussian(img, 5)
    save_result(result)


