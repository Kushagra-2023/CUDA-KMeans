import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def find_distances(points, centroids):
    return ((points - centroids[:, None, :]) ** 2).sum(dim=-1)

def get_new_centroids(points, labels, n, k):
    dim = points.shape[1]
    sum_points = torch.zeros((k, dim), device=device)
    counts = torch.zeros(k, device=device) + 1e-6
    for i in range(k):
        mask = labels == i
        sum_points[i] = points[mask].sum(dim=0)
        counts[i] = mask.sum()
    return sum_points / counts.unsqueeze(1)

def custom_kmeans_predict(points, k, tol=1e-8, max_iter=500):
    n = points.shape[0]
    indices = torch.linspace(0, n - 1, steps=k, device=points.device).long()
    centroids = points[indices]
    for _ in range(max_iter):
        distances = find_distances(points, centroids)
        labels = distances.argmin(dim=0)
        new_centroids = get_new_centroids(points, labels, n, k)
        shift = torch.norm(new_centroids - centroids)
        if shift < tol:
            break
        centroids = new_centroids
    return labels.cpu().numpy(), centroids.cpu().numpy()
