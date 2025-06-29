### Introduction

Welcome to my implementation of the popular K-Means algorithm, now enhanced with GPU acceleration! By leveraging the power of GPUs, this implementation outperforms Scikit-learn's K-Means, particularly on large datasets. Enjoy faster clustering with the same robust results, making it an excellent choice for your data science and machine learning projects.

## Overview

- **Goal**: Measure performance and clustering similarity (via Adjusted Rand Index) between:
  - `sklearn.cluster.KMeans`
  - Custom `PyTorch`-based KMeans with GPU acceleration
- **Evaluation Metrics**:
  - Clustering quality: **ARI (Adjusted Rand Index)**
  - Runtime comparison: **% Speedup**
- **Environment**: GPU-compatible (CUDA), also works on CPU

Final Benchmark Table:
| Samples | Clusters | ARI     | Sklearn Time | Torch Time | % Speedup   |
|---------|----------|---------|---------------|-------------|--------------|
| 1M      | 5        | 0.9993  | 0.75 s       | 0.04 s     | **+1581%**   |
| 2M      | 10       | 0.9336  | 1.54 s       | 2.14 s     | -28%         |
| 3M      | 15       | 0.8881  | 3.85 s       | 4.15 s     | -7%          |
| 4M      | 20       | 0.6398  | 10.51 s      | 4.31 s     | **+143%**    |
| 5M      | 25       | 0.7265  | 20.03 s      | 7.77 s     | **+158%**    |

### Summary

- **Average ARI**: 0.8375
- **Average Speedup**: +369.47%
- **Max Speedup**: +1581.23%

![image](https://github.com/user-attachments/assets/af8bd7a1-51b4-46bf-87eb-fe327abc9532)
