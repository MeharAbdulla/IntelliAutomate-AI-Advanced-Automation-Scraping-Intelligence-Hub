# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic dataset
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Plot dataset
plt.scatter(X[:, 0], X[:, 1], s=30, cmap='viridis')
plt.title("Generated Data")
plt.show()

# --------------------------
# Part 1: K-Means From Scratch
# --------------------------

def initialize_centroids(X, k):
    """Randomly initialize centroids."""
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def compute_distances(X, centroids):
    """Compute distances from each point to each centroid"""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return distances

def assign_clusters(distances):
    """Assign each point to the nearest centroid"""
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    """Update centroids based on cluster assignments"""
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        points = X[labels == i]
        if len(points) > 0:
            new_centroids[i] = np.mean(points, axis=0)
        else:
            new_centroids[i] = X[np.random.choice(X.shape[0])]
    return new_centroids

def k_means(X, k, max_iters=1000):
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        distances = compute_distances(X, centroids)
        labels = assign_clusters(distances)
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Run manual K-Means
k = 4
centroids, labels = k_means(X, k)

# Plot result
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
plt.title("K-Means From Scratch")
plt.show()

# --------------------------
# Part 2: K-Medoids From Scratch
# --------------------------

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b), axis=1)

def initialize_medoids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def assign_medoids(X, medoids):
    distances = np.array([[np.linalg.norm(x - m) for m in medoids] for x in X])
    return np.argmin(distances, axis=1)

def update_medoids(X, labels, k):
    new_medoids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            # Handle empty clusters by randomly choosing a new medoid
            new_medoids.append(X[np.random.choice(X.shape[0])])
            continue
        distances = np.sum(np.abs(cluster_points[:, np.newaxis] - cluster_points), axis=2)
        medoid_index = np.argmin(np.sum(distances, axis=1))
        new_medoids.append(cluster_points[medoid_index])
    return np.array(new_medoids)

def k_medoids(X, k, max_iters=1000):
    medoids = initialize_medoids(X, k)
    for _ in range(max_iters):
        labels = assign_medoids(X, medoids)
        new_medoids = update_medoids(X, labels, k)
        if np.allclose(medoids, new_medoids):
            break
        medoids = new_medoids
    return medoids, labels

# Run K-Medoids
medoids, medoid_labels = k_medoids(X, k)

# Plot K-Medoids result
plt.scatter(X[:, 0], X[:, 1], c=medoid_labels, s=30, cmap='viridis')
plt.scatter(medoids[:, 0], medoids[:, 1], s=300, c='red', marker='D')
plt.title("K-Medoids From Scratch")
plt.show()

# --------------------------
# Part 3: KMeans with scikit-learn
# --------------------------

kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x')
plt.title("K-Means using Scikit-Learn")
plt.show()

print("Inertia (Scikit-Learn):", kmeans.inertia_)

# --------------------------
# Part 4: Compare and Discuss
# --------------------------

print("\n--- Comparison and Discussion ---")
print("K-Means uses the average (mean) of points in a cluster as the center.")
print("K-Medoids chooses actual data points (medoids) as centers, which can make it more robust to outliers.")
print("You may notice that K-Medoids clusters might look slightly different, especially when clusters have noise or outliers.")

# --------------------------
# Part 5: Answer Questions
# --------------------------

print("\n--- Q&A ---")

# Q1: Which method is less sensitive to outliers?
print("Q1: Which method (K-Means vs K-Medoids) is less sensitive to outliers? Why?")
print("A1: K-Medoids is less sensitive to outliers because it selects actual data points as centers.")
print("Outliers have less influence on medoids, unlike K-Means which averages them into centroids.")

# Q2: What happens if you increase the number of clusters K too much?
print("\nQ2: What happens if you increase the number of clusters K too much?")
print("A2: Increasing K too much can lead to overfitting: each point becomes its own cluster, reducing generalization.")
print("Inertia keeps dropping, but the clusters lose meaning and may not represent natural groupings.")

# Q3: How important is the initialization of centroids/medoids?
print("\nQ3: How important is the initialization of centroids/medoids?")
print("A3: Initialization is very important.")
print("Poor initialization can lead to suboptimal clustering (bad local minima).")
print("That's why algorithms like KMeans++ exist to improve the starting centroids.")
