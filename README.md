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
 n_samples  n_clusters      ARI  sklearn_time  torch_time  percent_speedup
   1000000           5 0.999293      0.753174    0.044799      1581.234699
   2000000          10 0.933610      1.543428    2.144142       -28.016510
   3000000          15 0.888117      3.847941    4.146469        -7.199581
   4000000          20 0.639819     10.512112    4.314519       143.645044
   5000000          25 0.726543     20.032243    7.773595       157.696014

Summary:
Average ARI: 0.8375
Average % Speedup: 369.47%
Max % Speedup: 1581.23%

![image](https://github.com/user-attachments/assets/af8bd7a1-51b4-46bf-87eb-fe327abc9532)
