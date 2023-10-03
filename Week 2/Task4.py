import numpy as np

def initialize_centroids(X, k):
    # Initialize cluster centers by selecting k data points from X
    centroids_indices = np.random.choice(len(X), k, replace=False)
    centroids = X[centroids_indices]
    return centroids

def assign_to_clusters(X, centroids):
    # Assign each data point to the nearest cluster centroid
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments

def update_centroids(X, cluster_assignments, k):
    # Update cluster centroids as the mean of data points in each cluster
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[cluster_assignments == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
    return centroids

def k_means(X, k, max_iterations=100):
    # Initialize centroids
    centroids = initialize_centroids(X, k)

    for iteration in range(max_iterations):
        # Assign data points to clusters
        cluster_assignments = assign_to_clusters(X, centroids)

        # Update cluster centroids
        new_centroids = update_centroids(X, cluster_assignments, k)

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return cluster_assignments, centroids

# Example usage:
if __name__ == "__main__":
    # Generate sample data (5 blobs with 50 points each)
    np.random.seed(0)
    blobs = [np.random.randn(50, 2) + np.array([i * 3, i * 3]) for i in range(5)]
    data = np.vstack(blobs)

    # Number of clusters
    k = 5

    # Run K-Means
    cluster_assignments, cluster_centers = k_means(data, k)

    # Print cluster assignments and cluster centers
    print("Cluster Assignments:")
    print(cluster_assignments)
    print("\nCluster Centers:")
    print(cluster_centers)
