
## Introduction

 The optimization technique for vector databases in ANN search algorithms, specially with aim to enhance HNSW (Hierarchical Navigable Small World) algorithm. The proposed method employ Locally-adaptive Vector Quantization (LVQ) to compress individual vector sizes, and as a result, significantly reduces memory overhead. Building on conventional practices, the paper integrates optimization technique and evaluates them on the Sift dataset. Theoretical analysis and experimental results demonstrate that the proposed method achieves noticeable performance improvements in memory efficiency with only limited precision loss for TOP1 vector retrieval, and meanwhile partially reducing time complexity.


## Implementation

	HNSW Implementation
Each node appears at the bottom. For each newly inserted node, algorithm randomly decides whether it should take place in the upper layer.  The whole process repeats itself until the node is no longer picked in a certain layer. In each layer, the new node links with the existing nodes in that layer and forms a small-world graph structure. 
 
Index construction (inserting new vectors)as shown in
Algorithm pseudocode is shown in the below screenshot: First, we need to decide which layer to insert the new vector. Starting from the top layer, it performs a search-like process at each layer, looking for suitable neighboring points for connection. Meanwhile, it may need to delete a few existing connections to keep the structural properties of the graph.


```
Insert(Object &q, float *mean) {
    maxLyer = layer_edgeLists_.size() - 1;
    select a random layer for l
    for i=maxLyer to l-1 
    t=SearchLayer (q, ep, M, 1, i)[0]  
    ep=closest elements from t
    for i=min(maxLyer,l) to 0
    t=SearchLayer (q, ep, M, efConstruction, i) 
    select best M elements from tempRes
    connect best M elements from t to q 
    clear connected elements 
    enterPoints=closest elements from tempRes 
    if (level> maxLayer) 
    update the enterpoint
}
```

Index query
Algorithm pseudocode is shown below: The search picks a random starting point from the top layer and starts searching in the layer for the local nearest neighbor using greedy algorithm and then skips to the next layer.  The process repeats until it arrives at layer 0 and begins the real search there, moving continuously to the neighbor closer to the target.  If it is not able to find the closer point in the current layer, it will lower to the next one and continue the searching process until it reaches to the bottom and gets the result.

```
Insert(Object &q, float *mean) {
    maxLyer = layer_edgeLists_.size() - 1;
    select a random layer for l
    for i=maxLyer to l-1 
        t=SearchLayer (q, ep, M, 1, i)[0]  
        ep=closest elements from t
    for i=min(maxLyer,l) to 0
        t=SearchLayer (q, ep, M, efConstruction, i) 
        select best M elements from tempRes
        connect best M elements from t to q 
        clear connected elements 
        enterPoints=closest elements from tempRes 
        if (level> maxLayer) 
           update the enterpoint  

```

Index query
Algorithm pseudocode is shown below: The search picks a random starting point from the top layer and starts searching in the layer for the local nearest neighbor using greedy algorithm and then skips to the next layer.  The process repeats until it arrives at layer 0 and begins the real search there, moving continuously to the neighbor closer to the target.  If it is not able to find the closer point in the current layer, it will lower to the next one and continue the searching process until it reaches to the bottom and gets the result.

```
SearchLayer (Object&q, int ep, int ef, int lc) 
    Set [Object] is_visited
    priority_queue [Object] candidates 
    while(all elements in candidates)
       object c =candidates.top() 
       candidates.pop()
       if d(c,q)>d(result.top(),q)  break 
       for_each object e from friendlayers of c 
       if is_visited.find(e) != is_visited.end()
          add e to is_visited 
          if d(e, q)< d(result.top(),q) or result.size()<ef 
             add e to candidates
          if result.size()>ef 
             result.pop()
    return k nearest elements from the result


```
Pseudocode for the searching part
```
KNNSearch(Object &q, int K) {
    maxLyer = size of layer_edgeLists_-1;
    ep = enter_node_;
    for l=maxLyer to 1
        ep = SearchLayer(q, ep, 1, l)[0];
    return SearchLayer(q, ep, K, 0);
}

```


    LVQ Implementation

Aiming to convert vectors from floating-point numbers to compressed integers by the following quantization technique to reduce time complexity and memery consumption. 
The compressed vector structure is constructed as shown below.


```
Δ*⌊︃ (x- MINi)/ Δ+0.5⌋︃+MINi,    Δ= (MAXi- MINi)/255
```

```C++
struct LVQData {
	float scale_, bias_;
	int8_t compress_vec_[130];
};
```
```C++
    float lower = std::numeric_limits<float>::max();
    float upper = -std::numeric_limits<float>::max();
    for (int j = 0; j < dim; ++j) {
        auto x = static_cast<float>(src[j] - mean_[j]);
        lower = std::min(lower, x);
        upper = std::max(upper, x);
    }
    float scale = (upper - lower) / 255;
    float bias = lower - std::numeric_limits<int8_t>::min() * scale;
    if (scale == 0) {
        std::fill(dest.compress_vec_, dest.compress_vec_ + dim, 0);
    } else {
        float scale_inv = 1 / scale;
        for (int j = 0; j < dim; ++j) {
            auto c = std::floor((src[j] - mean_[j] - bias) * scale_inv + 0.5);
            dest.compress_vec_[j] = c;
        }
    }
```


    
### Experimental Results


                            | Plain HNSW | Optimal HNSW
    Dataset                 | SIFT_10K   | SIFT_10K
    Dimension               | 128        | 128   
    Number of Vectors       | 10000      | 10000
    Number of Queries       | 100	     | 100
    Distance	        | L2         |	L2
    Recall	                | 100%       | 98%
    Average Sum Latency(s)  | 1.80       |	1.27

### Compile
Runtime environment：Ubuntu 23.04，16G Memory，AMD Ryzen 7 5800H CPU，Compiler: Clang 18
```
cmake ..
```
## Improvements

In the future, in-memory pre-storage of large-scale vectors should be implemented, so as to adapt to SIFT_1M and draw more rigorous conclusions.

The practice of residual quantization is also one of the directions for improvements , and we believe that 98% recall for 10k vectors is far from the ultimate target of the project.
