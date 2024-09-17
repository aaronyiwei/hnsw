#include "hnsw.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

std::vector<int> HNSWGraph::SearchLayer(Point& q, int ep, int ef, int lc) {
    std::set<std::pair<double, int>> candidates;
    std::set<std::pair<double, int>> nearest_neighbors;
    std::unordered_set<int> is_visited;

    double td = q.dist(items_[ep]);
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
            td = q.dist(items_[ed]);
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

std::vector<int> HNSWGraph::KNNSearch(Point& q, int K) {
    int maxLyer = layer_edgeLists_.size() - 1;
    int ep = enter_node_;
    for (int l = maxLyer; l >= 1; l--) ep = SearchLayer(q, ep, 1, l)[0];
    return SearchLayer(q, ep, K, 0);
}

void HNSWGraph::AddEdge(int st, int ed, int lc) {
    if (st == ed) return;
    layer_edgeLists_[lc][st].push_back(ed);
    layer_edgeLists_[lc][ed].push_back(st);
}

void HNSWGraph::Insert(Point& q) {
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
    for (int i = maxLyer; i > l; i--) ep = SearchLayer(q, ep, 1, i)[0];
    for (int i = std::min(l, maxLyer); i >= 0; i--) {
        int MM = l == 0 ? MMax0_ : MMax_;
        std::vector<int> neighbors = SearchLayer(q, ep, ef_construction_, i);
        std::vector<int> selected_neighbors = std::vector<int>(neighbors.begin(), neighbors.begin()+std::min(int(neighbors.size()), M_));
        for (int n: selected_neighbors) AddEdge(n, nid, i);
        for (int n: selected_neighbors) {
            if (layer_edgeLists_[i][n].size() > MM) {
                std::vector<std::pair<double, int>> dist_pairs;
                for (int nn: layer_edgeLists_[i][n]) dist_pairs.emplace_back(items_[n].dist(items_[nn]), nn);
                std::sort(dist_pairs.begin(), dist_pairs.end());
                layer_edgeLists_[i][n].clear();
                for (int d = 0; d < std::min(int(dist_pairs.size()), MM); d++) layer_edgeLists_[i][n].push_back(dist_pairs[d].second);
            }
        }
        ep = selected_neighbors[0];
    }
    if (l == layer_edgeLists_.size() - 1) enter_node_ = nid;
}
