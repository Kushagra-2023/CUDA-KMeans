from main import custom_kmeans_predict
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score
import torch, time, pandas as pd
import matplotlib.pyplot as plt

def benchmark_kmeans(configs, device):
    results = []
    for n_samples, n_clusters in configs:
        print(f"\nRunning for n_samples={n_samples}, n_clusters={n_clusters}")
        X_np, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=2, random_state=42)
        X_torch = torch.tensor(X_np, dtype=torch.float32).to(device)

        start = time.time()
        sk_kmeans = SklearnKMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        sk_labels = sk_kmeans.fit_predict(X_np)
        sk_time = time.time() - start

        start = time.time()
        torch_labels, _ = custom_kmeans_predict(X_torch, k=n_clusters)
        torch_time = time.time() - start

        ari = adjusted_rand_score(sk_labels, torch_labels)
        percent_speedup = ((sk_time / torch_time) - 1) * 100
        results.append((n_samples, n_clusters, ari, sk_time, torch_time, percent_speedup))

    df = pd.DataFrame(results, columns=[
        "n_samples", "n_clusters", "ARI", "sklearn_time", "torch_time", "percent_speedup"
    ])

    print("\n📊 Final Benchmark Table:")
    print(df.to_string(index=False))

    print("\n📈 Summary:")
    print(f"Average ARI: {df['ARI'].mean():.4f}")
    print(f"Average % Speedup: {df['percent_speedup'].mean():.2f}%")
    print(f"Max % Speedup: {df['percent_speedup'].max():.2f}%")

    # plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df["n_samples"], df["percent_speedup"], marker='o', color='green')
    plt.title("PyTorch Speedup over Sklearn (%)")
    plt.xlabel("Number of Samples")
    plt.ylabel("Speedup (%)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df["n_samples"], df["ARI"], marker='o', color='blue')
    plt.title("ARI vs Number of Samples")
    plt.xlabel("Number of Samples")
    plt.ylabel("Adjusted Rand Index (ARI)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
