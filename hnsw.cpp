#include "hnsw.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>
void DecompressTo(double *dest, LVQData& q, const double *mean){
    for(int i = 0; i < 128; ++i) {
        dest[i] = q.scale_ * q.compress_vec_[i] + q.bias_ + mean[i];
    }
}
double Dist(double* point, double* other){
    double result = 0.0;
    for (int i = 0; i < 128; i++) result += (point[i] - other[i]) * (point[i] - other[i]);
    return result;
}
std::vector<int> HNSWGraph::SearchLayer(LVQData& q, int ep, int ef, int lc, double* mean, double* qmean) {
    std::set<std::pair<double, int>> candidates;
    std::set<std::pair<double, int>> nearest_neighbors;
    std::unordered_set<int> is_visited;
    double decom_vec[130],decom_vec2[130];
    DecompressTo(decom_vec, q, qmean);
    DecompressTo(decom_vec2, items_[ep], mean);
    double td = Dist(decom_vec, decom_vec2);
    candidates.insert(std::make_pair(td, ep));
    nearest_neighbors.insert(std::make_pair(td, ep));
    is_visited.insert(ep);
    while (!candidates.empty()) {
        auto ci = candidates.begin();
        candidates.erase(candidates.begin());
        int nid = ci->second;
        auto fi = nearest_neighbors.end();
        fi--;
        if (ci->first > fi->first) break;
        for (int ed: layer_edgeLists_[lc][nid]) {
            if (is_visited.find(ed) != is_visited.end()) continue;
            fi = nearest_neighbors.end();
            fi--;
            is_visited.insert(ed);
            DecompressTo(decom_vec2, items_[ed], mean);
            
            td = Dist(decom_vec, decom_vec2);
            if ((td < fi->first) || nearest_neighbors.size() < ef) {
                candidates.insert(std::make_pair(td, ed));
                nearest_neighbors.insert(std::make_pair(td, ed));
                if (nearest_neighbors.size() > ef) nearest_neighbors.erase(fi);
            }
        }
    }
    std::vector<int> results;
    for(auto &p: nearest_neighbors) results.push_back(p.second);
    return results;
}

std::vector<int> HNSWGraph::KNNSearch(LVQData& q, int K, double* mean, double* qmean) {
    int maxLyer = layer_edgeLists_.size() - 1;
    int ep = enter_node_;
    for (int l = maxLyer; l >= 1; l--) ep = SearchLayer(q, ep, 1, l, mean, qmean)[0];
    return SearchLayer(q, ep, K, 0, mean, qmean);
}

void HNSWGraph::AddEdge(int st, int ed, int lc) {
    if (st == ed) return;
    layer_edgeLists_[lc][st].push_back(ed);
    layer_edgeLists_[lc][ed].push_back(st);
}

void HNSWGraph::Insert(LVQData& q, double* mean) {
    int nid = items_.size();
    item_num_++;
    items_.push_back(q);
    // sample layer
    int maxLyer = layer_edgeLists_.size() - 1;
    int l = 0;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    while(l < ml_ && (1.0 / ml_ <= distribution(generator))) {
        l++;
        if (layer_edgeLists_.size() <= l) layer_edgeLists_.push_back(std::unordered_map<int, std::vector<int>>());
    }
    if (nid == 0) {
        enter_node_ = nid;
        return;
    }
    // search up layer entrance
    int ep = enter_node_;
    double decom_vec[130],decom_vec2[130];
    for (int i = maxLyer; i > l; i--) ep = SearchLayer(q, ep, 1, i, mean, mean)[0];
    for (int i = std::min(l, maxLyer); i >= 0; i--) {
        int MM = l == 0 ? MMax0_ : MMax_;
        std::vector<int> neighbors = SearchLayer(q, ep, ef_construction_, i, mean, mean);
        std::vector<int> selected_neighbors = std::vector<int>(neighbors.begin(), neighbors.begin()+std::min(int(neighbors.size()), M_));
        for (int n: selected_neighbors) AddEdge(n, nid, i);
        for (int n: selected_neighbors) {
            if (layer_edgeLists_[i][n].size() > MM) {
                std::vector<std::pair<double, int>> dist_pairs;
                DecompressTo(decom_vec, items_[n], mean);
                for (int nn: layer_edgeLists_[i][n]){
                    DecompressTo(decom_vec2, items_[nn], mean);
                    dist_pairs.emplace_back(Dist(decom_vec,decom_vec2), nn);
                }
                std::sort(dist_pairs.begin(), dist_pairs.end());
                layer_edgeLists_[i][n].clear();
                for (int d = 0; d < std::min(int(dist_pairs.size()), MM); d++) layer_edgeLists_[i][n].push_back(dist_pairs[d].second);
            }
        }
        ep = selected_neighbors[0];
    }
    if (l == layer_edgeLists_.size() - 1) enter_node_ = nid;
}
