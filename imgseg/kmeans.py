'''
Created on 14 Nov 2016

@author: Fabian Meyer
'''

import numpy as np
import math

_CLUSTER_COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

def _distance(pixelA, pixelB):
    assert(len(pixelA) >= 3 and len(pixelB) >= 3)
    dist = np.float64(0)
    for i in range(3):
        dist += math.pow(float(pixelA[i]) - float(pixelB[i]), 2)

    return dist

def _assign_clusters(img, pixels, clusters, k):
    # assign pixels to clusters
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # find minimal _distance for this pixel
            min_dist = (-1, 0)
            for i in range(k):
                dist = _distance(img[y, x], clusters[i])
                if min_dist[0] < 0 or dist < min_dist[1]:
                    min_dist = (i, dist)
            pixels[y, x] = min_dist[0]

def _calc_clusters_mean(img, pixels, clusters_next, clusters_count, k):
    # calculate rgb sum per cluster
    clusters_count.fill(0)
    clusters_next.fill(0)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            idx = pixels[y, x]
            clusters_count[idx] += 1
            for i in range(3):
                clusters_next[idx][i] += img[y, x][i]

    # calculate rgb mean per cluster
    for i in range(k):
        for b in range(3):
            clusters_next[i][b] /= clusters_count[i]

def _move_cluster_center(clusters, clusters_next, k):
    # move cluster center and check if it has changed
    cluster_moved = False
    for i in range(k):
        for b in range(3):
            if clusters_next[i][b] != clusters[i][b]:
                    cluster_moved = True

            clusters[i][b] = np.uint8(clusters_next[i][b])

    return cluster_moved

def kmeans(img, k):
    clusters = np.empty((k, 3), dtype=np.uint8)
    for i in range(k):
        clusters[i][:] = np.random.randint(0, 255, 3, dtype=np.uint8)

    clusters_count = np.zeros(k, dtype=np.int)
    clusters_next = np.zeros((k, 3), dtype=np.uint64)
    pixels = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
    cluster_moved = True
    while cluster_moved:
        _assign_clusters(img, pixels, clusters, k)
        _calc_clusters_mean(img, pixels, clusters_next, clusters_count, k)
        print('---------------')
        print(clusters)
        print(clusters_next)
        cluster_moved = _move_cluster_center(clusters, clusters_next, k)

    result = np.empty((img.shape[0], img.shape[1] , 3), dtype=np.uint8)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            result[y, x] = _CLUSTER_COLORS[pixels[y, x]]
    return result
