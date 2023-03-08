# %pylab inline
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def clustering_kmeans(df):
    
    ###############
    # Adapted grid search 
    ###############
    grid_search = {}
    for nb_clusters in range(2,10,1):
        model_clustering = KMeans(n_clusters= nb_clusters, n_init=5, init='k-means++', verbose=0, random_state=1).fit(df)
        target_pred = pd.Series(model_clustering.labels_, name='Elec_demand_cluster')
        # Or model_clustering.inertia_ = sse which corresponds to Elbow Method 
        grid_search[nb_clusters] = silhouette_score(df, target_pred, metric="euclidean")
    
    
    plt.plot(grid_search.keys(), grid_search.values(),'-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette criterion to maximize')
    plt.title('Clustering performance over whole dataset')
    plt.show()
       
    ###############
    # Optimal model
    ###############
    nb_clusters = max(grid_search, key=grid_search.get)
    score = grid_search[nb_clusters]
    model_clustering = KMeans(n_clusters= nb_clusters, n_init=5, init='k-means++', random_state=1).fit(df)
    target_pred = pd.Series(model_clustering.labels_, name='Elec_demand_cluster')
    centroids = pd.DataFrame(model_clustering.cluster_centers_)
    
    
    ##################
    # Performance generalization
    ##################
    # print("Adjusted Rand Index: %.3f" % adjusted_rand_score(labels_true=target, labels_pred=target_pred))
    # print("\nClustering : silhouette score : %.3f" % silhouette_score(data_std, target_pred, metric="euclidean"))
	
    return model_clustering, target_pred, nb_clusters, score, centroids
