import numpy as np

def periodic_dis(x,y,n_vars_per):
    dist_sq = 0.
    for i in range(n_vars_per):
        dist_sq += (min(np.abs(x[i]-y[i]),2*np.pi - np.abs(x[i]-y[i])))**2
    for i in range(n_vars_per,len(x),1):
        dist_sq += (x[i]-y[i])**2
    return dist_sq

def assign_label(x, centroids, n_vars_per):
    distances = []
    for i in range(len(centroids)):
        c = centroids[i,:]
        distances.append(periodic_dis(c,x,n_vars_per))
    return np.argmin(distances)

def convergence(new_centroids, centroids,n_vars_per):
    displacement = 0.
    n_clusters = len(centroids)
    for i in range(n_clusters):
        displacement += periodic_dis(new_centroids[i,:],centroids[i,:],n_vars_per)
    displacement = np.sqrt(displacement/float(n_clusters)/float(len(centroids[0])))
    print('Displacement of centroids: {}'.format(displacement))
    return displacement

def myInertia(centroids, data, n_vars_per):
    total_inertia = 0.
    clusters = np.zeros(len(centroids))
    for row in data:
        centroid_idx = assign_label(row,centroids,n_vars_per)
        clusters[centroid_idx] += periodic_dis(row, centroids[centroid_idx],n_vars_per)

    total_inertia = clusters.sum()
    return total_inertia

def periodic_mean(data):
    n = len(data)
    cart_coords = np.zeros((n,2))
    for i in range(len(data)):
        cart_coords[i,0] = np.cos(data[i])
        cart_coords[i,1] = np.sin(data[i])
    # print(cart_coords)
    mean_vector = np.mean(cart_coords,axis=0)
    # print('Mean vector: {}'.format(mean_vector))
    return np.arctan2(mean_vector[1],mean_vector[0])
