<p align="center">
 <b> Optimized implementation of HNSW with LVQ approach </b>
</p>

## Introduction

 The optimization technique for vector databases in ANN search algorithms, specially with aim to enhance HNSW (Hierarchical Navigable Small World) algorithm. The proposed method employ Locally-adaptive Vector Quantization (LVQ) to compress individual vector sizes, and as a result, significantly reduces memory overhead. Building on conventional practices, the paper integrates optimization technique and evaluates them on the Sift dataset. Theoretical analysis and experimental results demonstrate that the proposed method achieves noticeable performance improvements in memory efficiency with only limited precision loss for TOP1 vector retrieval, and meanwhile partially reducing time complexity.


## Usage

    | Plain HNSW | Optimal HNSW
 ---- | ----- | ------  
 Dataset  | SIFT_10K | SIFT_10K
 Dimension  | 128 | 128   
 Number of Vectors | 10000 | 10000
 Number of Queries	| 100	| 100
 Distance	L2 |	L2
 Recall	100% | 	98%
 Average Sum Latency	1.80(s) |	1.27(s)


## 

## Improvements
