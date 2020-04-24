from heapq import heappop, heappush

import numpy as np
from scipy.spatial.distance import cdist


def pq(data, P, init_centroids, max_iter):
    def KMeans(data, centroid):
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
            # Measure the distance to every centroid loacation in codebook
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

    data_partitioned = np.hsplit(data, P)

    codebooks = []
    codes = []

    for i in range(P):
        W, C = KMeans(data_partitioned[i], init_centroids[i])
        codebooks.append(W)
        codes.append(C)
    codebooks = np.asarray(codebooks)
    codes = np.asarray(codes, dtype='uint8').T

    return codebooks, codes


def query(queries, codebooks, codes, T):
    def pqTable_func(query, codebooks):
        P, K, D = codebooks.shape
        pqTable = dict()
        for i in range(P):
            d = cdist(query[i].reshape(-1, D), codebooks[i], 'cityblock')
            c_i = np.argsort(d)
            d = np.sort(d)
            qvsU = {}
            for k in range(K):
                qvsU[k] = (d[0][k], c_i[0][k])
            pqTable[i] = qvsU
        return pqTable

    def pqTable_func(query):
        pqTable = dict()
        for i in range(P):
            d = cdist(query[i].reshape(-1, D), codebooks[i], 'cityblock')
            c_i = np.argsort(d)
            d = np.sort(d)
            qvsU = {}
            for k in range(K):
                qvsU[k] = (d[0][k], c_i[0][k])
            pqTable[i] = qvsU
        return pqTable

    def traversed(L, q_i, T, K):
        if L > T:
            return False
        for i in range(len(q_i)):
            if q_i[i] >= K:
                return False
        return True

    def getvalue(q_i, pqTable):
        P = len(q_i)
        dist = 0
        c_i = []
        for p in range(P):
            qvsU = pqTable[p]
            d, i = qvsU[q_i[p]]
            dist += d

            c_i.append(i)

        return (dist, tuple(q_i), tuple(c_i))

    def label(c_i, out):
        if out == None:
            out = set()
        val = indexdict.get(c_i)
        if val:
            for i in val:
                out.add(i)
        return out

    def createIndex(codes):
        indexdict = {}

        for i in range(len(codes)):
            temp = indexdict.get(tuple(codes[i]))
            if None == temp:
                val = []
                val.append(i)
                indexdict[tuple(codes[i])] = val
            else:
                val.append(i)
                indexdict[tuple(codes[i])] = val

        return indexdict

    def querysearch(query):
        pqTable = pqTable_func(query)
        out = set()
        trav = {}
        q_i = [0 for x in range(P)]

        trav[tuple(q_i)] = True
        h = []
        (dist, q_i, c_i) = getvalue(q_i, pqTable)
        heappush(h, (dist, q_i, c_i))
        L = 0 if (out == set()) else (len(out))
        while L < T and len(h) > 0:
            dist, q_i, c_i = heappop(h)
            out = (label(c_i, out))
            L = 0 if (out == set()) else (len(out))
            for p in range(P):
                if (q_i[p] < K - 1 and (
                        tuple(np.subtract(q_i, Id[p]) == tuple([0 for x in range(P)])) or trav.get(
                    np.add(q_i, Id[p])))):
                    (dist, q_i, c_i) = getvalue(np.add(q_i, Id[p]), pqTable)
                    heappush(h, (dist, q_i, c_i))

        return out

    P, K, D = codebooks.shape
    nQ = queries.shape[0]
    indexdict = createIndex(codes)
    Id = np.identity(P)
    qparts = np.stack(np.hsplit(queries, P), axis=1)
    CandidateList = []
    
    for q in range(nQ):
        Cset = querysearch(qparts[q])
        CandidateList.append(Cset)

    return CandidateList
