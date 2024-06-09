import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def find_distances(points, centroids):
    # Compute distances between points and centroids
    distances = ((points - centroids[:, None, :]) ** 2).sum(axis=-1)
    return distances

def get_new_centroids(points, indexes_for_clusters, n, k):
    # Initialize tensors to accumulate sum and count of points for each centroid
    centroids = torch.zeros((k, points.shape[1]), device=device, dtype=torch.float)
    sum_points = torch.zeros((k, points.shape[1]), device=device, dtype=torch.float)
    counts = torch.zeros(k, device=device, dtype=torch.float) + 1e-6

    for i in range(k):
        mask = (indexes_for_clusters == i)
        sum_points[i] = points[mask].sum(dim=0)
        counts[i] = mask.sum()

    # Compute new centroids
    centroids = sum_points / counts.unsqueeze(1)

    return centroids

'''
points: 2D torch.tensor of points in (x, y) format (ex. [[1, 2], [4,5], ..])
k: number of clusters in dataset
max_iter: maximum number of times the function checks for convergance.
'''
def kmeans(points, k, max_iter=50):
    points = points.to(device)
    n = points.shape[0]
    indexes = torch.linspace(0, n-1, steps=k, device=device).long()
    centroids = points[indexes]

    for iteration in range(max_iter):
        distances = find_distances(points, centroids)
        indexes_for_clusters = distances.argmin(axis=0)
        new_centroids = get_new_centroids(points, indexes_for_clusters, n, k)

        # Check for convergence
        if (((new_centroids-centroids) ** 2).sum())==0:
          break
        else:
          centroids = new_centroids

    print(f"Converged in {iteration+1} iterations")
    return new_centroids
