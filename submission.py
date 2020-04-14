def KMeans(data,centroid,max_iter):
    
    # std_dev = data.std(axis=0)
    # zero_std_mask = std_dev == 0
    # if zero_std_mask.any():
    #     std_dev[zero_std_mask] = 1.0  
    # data= data / std_dev
    
    # Number of clusters
    K = centroid.shape[0]
    
    thresh=1e-5
    
    # Number of training data
    N = data.shape[0]


    centroid_prev = np.zeros(centroid.shape) # to store previous centroids
    centroid_curr = np.array(centroid,copy=True) # Store current centroids

    clusters = np.zeros(N,dtype='uint8')
    distances = np.zeros((N,K))

    error = thresh+1
    
    iter=0

    while error >thresh and iter<max_iter:
        iter+=1
        # Measure the distance to every centroid
        for i in range(K):
            distances[:,i] = np.linalg.norm(data - centroid_curr[i], axis=1,ord=1)
        # Assign all training data to closest centroid
        clusters = np.argmin(distances, axis = 1)
        
        centroid_prev = np.array(centroid_curr,copy=True)
        # Calculate median for every cluster and update the centroid
        for i in range(K):
            centroid_curr[i] = np.nanmedian(data[clusters == i], axis=0)
        
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
