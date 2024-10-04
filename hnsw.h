#pragma once

#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>
struct LVQData {
    float scale_, bias_;
    std::pair<float, float> local_cache_;
    int8_t compress_vec_[130];
    // double mean[];
    // int8_t
};
struct Point {
    Point(std::vector<double> _values) : values(_values) {}
    std::vector<double> values;
    // Assume L2 distance
    double dist(Point &other) {

        double result = 0.0;
        for (int i = 0; i < values.size(); i++)
            result += (values[i] - other.values[i]) * (values[i] - other.values[i]);
        return result;
    }
};
struct LVQPoint {
    LVQPoint(std::vector<int8_t> _values) : values(_values) {}
    std::vector<int8_t> values;
    // Assume L2 distance
    double dist(Point &other) {
        double result = 0.0;
        for (int i = 0; i < values.size(); i++)
            result += (values[i] - other.values[i]) * (values[i] - other.values[i]);
        return result;
    }
};
struct HNSWGraph {
    HNSWGraph(int M, int MMax, int MMax0, int ef_construction, int ml)
        : M_(M), MMax_(MMax), MMax0_(MMax0), ef_construction_(ef_construction), ml_(ml) {
        layer_edgeLists_.push_back(std::unordered_map<int, std::vector<int>>());
    }

    // Number of neighbors
    int M_;
    // Max number of neighbors in layers >= 1
    int MMax_;
    // Max number of neighbors in layers 0
    int MMax0_;
    // Search numbers in construction
    int ef_construction_;
    // Max number of layers
    int ml_;

    // number of items
    int item_num_;
    // actual std::vector of the items
    std::vector<LVQData> items_;
    // adjacent edge lists in each layer
    std::vector<std::unordered_map<int, std::vector<int>>> layer_edgeLists_;
    // enter node id
    int enter_node_;

    std::default_random_engine generator;

    // methods
    void AddEdge(int st, int ed, int lc);
    std::vector<int> SearchLayer(LVQData &q, int ep, int ef, int lc, float *mean);
    void Insert(LVQData &q, float *mean);
    // void CompressTo()
    std::vector<int> KNNSearch(LVQData &q, int K, float *mean);

    void PrintGraph() {
        for (int l = 0; l < layer_edgeLists_.size(); l++) {
            std::cout << "Layer:" << l << std::endl;
            for (auto it = layer_edgeLists_[l].begin(); it != layer_edgeLists_[l].end(); ++it) {
                std::cout << it->first << ":";
                for (auto ed : it->second)
                    std::cout << ed << " ";
                std::cout << std::endl;
            }
        }
    }
};
