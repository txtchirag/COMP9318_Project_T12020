def KMeans(X,centers):
  # Number of clusters
  K = 256
  thresh=1e-5
  # Number of training data
  n = X.shape[0]
  # Number of features in the data
  c = X.shape[1]


  centers_old = np.zeros(centers.shape) # to store old centers
  centers_new = centers.copy() # Store new centers

  clusters = np.zeros(n)
  distances = np.zeros((n,K))

  error = np.linalg.norm(centers_new - centers_old,ord=1)

  # When, after an update, the estimate of that center stays the same, exit loop
  while error >thresh:
      # Measure the distance to every center
      for i in range(K):
          distances[:,i] = np.linalg.norm(X - centers_new[i], axis=1,ord=1)
      # Assign all training data to closest center
      clusters = np.argmin(distances, axis = 1)
      
      centers_old = deepcopy(centers_new)
      # Calculate mean for every cluster and update the center
      for i in range(K):
          centers_new[i] = np.median(X[clusters == i], axis=0)
      error = np.linalg.norm(centers_new - centers_old,ord=1)
  
  return np.asarray(centers_new),np.asarray(clusters,dtype='uint8')

def pq(data, P, init_centroids, max_iter):
    data_partitioned=np.asarray(np.hsplit(data,P))
    whitened=scipy.cluster.vq.whiten(data_partitioned)
    codebooks=[]
    codes=[]
    for i in range(P):
        W,C=KMeans(whitened[i],init_centroids[i])
        codebooks.append(W)
        codes.append(C)
    codebooks=np.asarray(codebooks)
    codes=np.asarray(codes,dtype='uint8').T
    print(codebooks.shape,codes.shape)
    return codebooks,codes

def query(queries, codebooks, codes, T):
    pass
