import numpy as np
import matplotlib.pyplot as plt

# Assign all constants values
DATASET_PATH = "./dataset" # Path of dataset file
MAX_CLUSTER_NUM = 9 # Maximum number of cluster
MAX_ITER = 100 # Maximum number of iteration 
RAND_SEED = 42 # Specific random seed number 
MODULE_NAME= "KMeansPlusPlus" # Declare module name to use as file output name

# Specific random seed
np.random.seed(RAND_SEED)

"""
    Create function to load dataset file
        Input: path: Str
        Output: dataset: List[List[Float]]
"""
def loadFile(path=DATASET_PATH) : 
    # separate dataset and labels
    dataset = []
    # read file with open function
    with open(path) as F:
        # read file line by line
        for line in F:
            # transform data into array 
            element = line.strip().split(' ')
            # add values to dataset
            dataset.append(element[1:])
    
    # return data as numpy array and transform data values into float type
    return np.array(dataset).astype(float)

"""
    Create function to compute distance by L2 norm (Euclidean) distance
        Input: point a: List[Float], b: List[Float] 
        Output: distance between 2 points : Float
"""
def computeDistance(a, b):
    # Use L2 norm in numpy to calculate distance
    return np.linalg.norm((a-b), ord=2)

"""
    Create function to initialise the centroid for KMeans++ algorithms

    Logic:
        For an object obj in the dataset X let R(obj) be the distance from obj to the closest cluster representative we have already chosen.

            1. Select one representative Y1 uniformly at random from X
            2. for i= 2, ... ,k 
                    Select representative Y from X with probability Yi = obj being equal to R(obj)^2/sum(R(obj)^2)
            3. Proceed with the standard k-means using Y1, ... , Yk as initial cluster representatives

    Input:  dataset x: List[List[Float]], the number of cluster k: Int
    Output: cluster: Dict[Int, Dict]
"""
def initialSelection(x, k):
    # initialise the cluster hashmap
    clusters = {}
    # Assign first centroid by random
    clusters[0] = { 
        "centroid" : x[np.random.choice(x.shape[0])],
        "points" : []
    }
    # Assign the rest of cluster by computeing the distance where k= 2, ..., k
    for i in range(1, k):
        # Initailise distances list 
        distances = []
        # interate dataset to compute the next centroid of each clusters
        for point in x:
            # Compute the distance of 'points' from each of the provious and selected the centroid by the minimum distance
            dist = min([ computeDistance(point, cluster['centroid']) for cluster in clusters.values() ])
            # Add to distances list
            distances.append(dist)

        # transform to numpy array
        distances = np.array(distances)
        # find the possibility values by Square(Dist)/sum(Square(Dist))
        probability = distances**2 / np.sum(distances**2)
        # assign new centroid by random with probability number
        new_centroid = x[np.random.choice(x.shape[0], p=probability)]

        # assign new cluster
        clusters[i] = {'centroid': new_centroid, 'points': []}
        
    # return the cluster
    return clusters


"""
    Create function to compute cluster representatives
        Input: clusters set C: Dict[int, dict] 
        Output: clusters set: Dict[int, dict] or None 
"""
def computeClusterRepresentatives(C):
    # Initailise to flag the improvement of representative points (False mean "no update", True is mean "update")
    is_improve = False
    # Initialise new clusters dictionary 
    new_cluster = {}
    
     # Iterate each clusters to update their centroid 
    for cluster_id, cluster in enumerate(C.values()):
        # Compute new center by average all distance of its cluster.
        new_center = np.average(np.array(cluster['points']), axis=0)

        # Flag the improvement value of if some center is updated.
        if not np.array_equal(new_center, cluster['centroid']):
            is_improve = True
        
        # Assign cluster obj to the new_cluster
        new_cluster[cluster_id] = {
            'centroid' : new_center,
            'points' : cluster['points']
        }
    
    # return value if there are some improvement
    if is_improve: 
        return new_cluster
    
"""
    Create function to assign datapoint into each clusters by centroids 
        Input: 
            dataset x: List[List[Float]], 
            the number of cluster k: Int,
            clusters set: Dict[int, Dict]
        Output: clusters: Dict[int, dict] or None 
"""
def assignClusterIds(x, k, Y):
    # Assign new cluster set as Y
    new_cluster = Y
    
    # Clear all previous assigned datapoint 
    for id in new_cluster.keys():
        new_cluster[id]['points'] = []

    # Iterate dataset to assign datapoints into each clusters
    for i in range(len(x)):
        # initialise distance for datapoint
        distances = []
        # assign current datapoint
        datapoint = x[i]
        # iterate each cluster to calculate distance  
        for i in range(k):
            # find distance with computeDisatance function between datapoint and centroid
            distance = computeDistance(datapoint, new_cluster[i]['centroid'])
            
            # Store all distances 
            distances.append(distance)
        
        # find local minimum to assign the cluster with cluster id
        cluster_id = np.argmin(distances)
        # Add datapoint to the cluster array
        new_cluster[cluster_id]['points'].append(datapoint)

    # return new cluster
    return new_cluster


"""
    Create function to do the KMeans process:
        Input: 
            dataset x: List[List[Float]], 
            the number of cluster k: Int,
            the maximum number of iteration maxIter= Int
        Output: clusters: Dict[int, dict] or None 
"""
# Create kMeans function to process e
def kMeansPlusPlus(x, k, maxIter):
    # Initialize represntative set of clusters 
    cluster_set = initialSelection(x , k)

    # Iterate to update clusters along with the maximum number of iterations
    for i in range(maxIter):
        # Create clusters (C1 .... Ck) by assigning each
            #point in D to closest representative in cluster set
            #using the distance function Dist(.,.);
        cluster_set = assignClusterIds(x, k, cluster_set)
        
        # Recreate set S by determining one representative cluster 
        updated_clusters = computeClusterRepresentatives(cluster_set)

        # Break the loop if a clusters set does not update (convergence)
        if not updated_clusters: 
            break;
        
        # update clusters before next iteration
        cluster_set = updated_clusters
        
    # return cluster
    return cluster_set


"""
    Create function to compute silhouette coefficient score:
        Input: clusters: Dict[Int, Dict], 
        Output: silhouette coefficient score: Float
"""
def computeSilhouttee(clusters):
    # Transform cluster dictionary into array List[Dict] (Easier to handle iteration)
    cluster_arr = np.array([ val for val in clusters.values()])
    # count cluster number
    n_cluster = len(cluster_arr)

    # Due to silhouette score can not be computed with 1 cluster so we return 0 as a score
    if n_cluster == 1: 
        return 0
    
    # Initialise the array of silhouette value for each element 
    silhouette_arr = []

    # Iterate cluster list to compute distance and silhoutte value in each cluster
    for cluster in cluster_arr:
        # Assign datapoint list of the current cluster
        points_arr = cluster['points']
        
        # iterate each point inside the cluster
        for point in points_arr:
            # Compute the summation of cohesive distances (distance between current point and points in the same cluster) 
            # Using computeDistance function as an assign function to compute distance
            cohesive_dist = np.sum([computeDistance(point, p) for p in points_arr if not np.array_equal(p, point)])
            # Compute Interdistance; cohesive distance divide by the number of points inside array
            # maximize the number to avoid crashing
            a_i = cohesive_dist / max(len(points_arr) - 1, 1)
            
            # Initialise seperation distances (distance outside the cluster)
            seperated_dist = []
            # Iterate cluster list to handle with outside cluster
            for other_cluster in cluster_arr:
                # only focus on the other clusters
                if not np.array_equal(other_cluster['centroid'], cluster['centroid']):
                    # Compute the sum of distance between current point and point from different cluster
                    dist = np.sum([computeDistance(point, p) for p in other_cluster['points']])
                    # Compute intradistance by divide the sum of distance by the number of other clusters
                    # maximize the number to avoid crashing
                    intra_dist = dist/ max(len(other_cluster['points']), 1)
                    # Add  intradistance to separation distances list
                    seperated_dist.append(intra_dist)
            
            # Minimise the separation distance list 
            b_i = np.min(seperated_dist)
            
            # Compute Silhouette values for position i by (Dout[min] - Din[avg]) / max{Dout[min], Din[avg]}
            silhouette_i = (b_i - a_i) / max(a_i, b_i)
            # Add Silhouette values to list 
            silhouette_arr.append(silhouette_i)

    # return Silhouette Coefficient score by averaging all values in the list
    return np.mean(silhouette_arr)


"""
    Create function to plot barchart between  the number of cluster and silhouette coefficient score:
        Input: 
            clusters: List[Dict[int, Dict]],
        Output: None [but have a generate file process]
"""
def plot_silhouttee(clusters):
    # Calculate silhouettes scores for each clusters
    silhouettes = [computeSilhouttee(clust) for clust in clusters]

    # Print the best silhoette score
    print(f"The greatest Silhouette Score is {np.max(silhouettes)} with {np.argmax(silhouettes) + 1} clusters")
    
    # config figure dpi to get a better quality 
    plt.figure(figsize=(12, 6), dpi=258)
    # Plot bar chart with X-axis as the number of cluster and Y-axis as s silhouette coeffiecient 
    plt.bar(range(1, len(silhouettes)+1), list(silhouettes), align="center" )
    # Customize label on  x and y axis
    plt.xlabel("Number of Cluster", fontsize=12)
    plt.ylabel("Silhouette Coefficient", fontsize=12)
    # Customize titile of graph
    plt.title(MODULE_NAME, fontsize=16)    
    
    # Save output picture to current directory 
    plt.savefig(f"{MODULE_NAME}_output.jpg")


if __name__ == "__main__" :
    print("\n ***** Start KMean++ Clustering ***** \n")

    # Load dataset from specific path 
    dataset = loadFile(DATASET_PATH)
    # Find clusters according to the maximum of cluster number : List[ Dict[Int,Dict] ]
    clusters = [ kMeansPlusPlus(dataset, k, MAX_ITER) for k in range(1, MAX_CLUSTER_NUM+1) ]
    # Plot Barchart between the number of clusters and Silhouette Coefficient Score
    plot_silhouttee(clusters)
    
    print("\n ***** KMean++ Clustering has been completed! *****")
