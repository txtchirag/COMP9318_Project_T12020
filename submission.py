import numpy as np
from scipy.spatial.distance import cdist


def KMeans(data, centroid, max_iter):
    # Number of clusters
    K = centroid.shape[0]
    # Number of training data
    N = data.shape[0]

    thresh = 1e-5
    error = thresh + 1

    iter = 0
    codebook_curr = np.array(centroid, copy=True)  # Store current centroids in codebook

    while error > thresh and iter < max_iter:
        iter += 1
        # Measure the distance to every centroid location in codebook
        distances = cdist(data, codebook_curr, 'cityblock')
        # Assign all training data to closest centroid in codes
        codes = np.argmin(distances, axis=1)

        codebook_prev = np.array(codebook_curr, copy=True)
        # Calculate median for every cluster in codes and update the centroid in codebook
        for c in range(K):
            points = data[codes == c]
            if points.size != 0:
                codebook_curr[c] = np.median(points, axis=0)

        error = np.linalg.norm(codebook_curr - codebook_prev) / np.sqrt(N)

    # Measure the distance to every centroid in codebook
    distances = cdist(data, codebook_curr, 'cityblock')
    # Assign all training data to closest centroid in codes
    codes = np.argmin(distances, axis=1)

    return np.asarray(codebook_curr), np.asarray(codes, dtype='uint8')


def pq(data, P, init_centroids, max_iter):
    data_partitioned = np.asarray(np.hsplit(data, P))

    codebooks = []
    codes = []

    for i in range(P):
        W, C = KMeans(data_partitioned[i], init_centroids[i], max_iter)
        codebooks.append(W)
        codes.append(C)
    codebooks = np.asarray(codebooks)
    codes = np.asarray(codes, dtype='uint8').T

    return codebooks, codes


def query(queries, codebooks, codes, T):
    # locate dictionary for each query vector and intersect each vector
    pass
