import torch
from eval import benchmark_kmeans

configs = [
    (1000000, 5),
    (2000000, 10),
    (3000000, 15),
    (4000000, 20),
    (5000000, 25)
]

device = "cuda" if torch.cuda.is_available() else "cpu"
benchmark_kmeans(configs, device)
