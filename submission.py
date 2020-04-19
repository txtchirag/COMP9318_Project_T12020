def KMeans(data,centroid,max_iter):
    # Number of clusters
    K = 256
    thresh=1e-5
    # Number of training data
    N = data.shape[0]
    # Number of features in the data
    M = data.shape[1]


    centroid_prev = np.zeros(centroid.shape) # to store previous centroids
    centroid_curr = np.array(centroid,copy=True) # Store current centroids

    clusters = np.zeros(N)
    prev_cluster=np.array(clusters,copy=True)
    distances = np.zeros((N,K))

    error = thresh +1
    iter=0

    while (error >thresh or clusters.all()!=prev_cluster.all()) and iter<max_iter:
        iter+=1
        # Measure the distance to every centroid
        for i in range(K):
            distances[:,i] = np.linalg.norm(data - centroid_curr[i], axis=1,ord=1)
        # Assign all training data to closest centroid
        prev_cluster=np.array(clusters,copy=True)
        clusters = np.argmin(distances, axis = 1)
        
        centroid_prev = np.array(centroid_curr,copy=True)
        # Calculate median for every cluster and update the centroid
        for i in range(K):
            centroid_curr[i] = np.median(data[clusters == i], axis=0)
        
        error = np.linalg.norm(centroid_curr - centroid_prev,ord=1)
    
    return np.asarray(centroid_curr),np.asarray(clusters,dtype='uint8')
  
def pq(data, P, init_centroids, max_iter):
    data_partitioned=np.asarray(np.hsplit(data,P))
    
    codebooks=[]
    codes=[]

    for i in range(P):
        W,C=KMeans(data_partitioned[i],init_centroids[i],max_iter)
        codebooks.append(W)
        codes.append(C)
    codebooks=np.asarray(codebooks)
    codes=np.asarray(codes,dtype='uint8').T
    
    return codebooks,codes
