#include "hnsw.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <utility>
#include <random>
#include <vector>
#include <math.h>
struct LVQData {
    float scale_,bias_;
    std::pair<double, double> local_cache_;
    int8_t compress_vec_[];
    //int8_t
};
std::pair<double, double> MakeLocalCache(const int8_t *c, double scale, double dim) {
        int norm1 = 0;
        int norm2 = 0;
        for (int i = 0; i < dim; ++i) {
            norm1 += c[i];
            norm2 += c[i] * c[i];
        }
        return {norm1 * scale, norm2 * scale * scale};
    }
 void CompressTo(const Point src, LVQData& dest, const double *mean_, int dim) const {

        //int8_t *compress = dest.compress_vec_;

        double lower = std::numeric_limits<double>::max();
        double upper = -std::numeric_limits<double>::max();
        for (int j = 0; j < dim; ++j) {
            auto x = static_cast<double>(src.values[j]-mean_[j]);
            lower = std::min(lower, x);
            upper = std::max(upper, x);
        }
        double scale = (upper - lower) / 255;
        double bias = lower - std::numeric_limits<double>::min() * scale;
        if (scale == 0) {
            std::fill(dest.compress_vec_, dest.compress_vec_ + dim, 0);
        } else {
            double scale_inv = 1 / scale;
            for (int j = 0; j < dim; ++j) {
                auto c = std::floor((src.values[j] - mean_[j] - bias) * scale_inv + 0.5);
                //assert(c <= std::numeric_limits<CompressType>::max() && c >= std::numeric_limits<CompressType>::min());
                dest.compress_vec_[j] = c;
            }
        }
        dest.scale_ = scale;
        dest.bias_ = bias;
        dest.local_cache_ = MakeLocalCache(dest.compress_vec_, scale, dim);
    }
void RandomTest(int numItems, int dim, int numQueries, int K) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    std::vector<Point> random_items;
    double new_mean[dim+1];
    int cur_vec_num = 0;
    for (int i = 0; i < numItems; i++) {
        std::vector<double> temp(dim);
        for (int d = 0; d < dim; d++) {
            temp[d] = distribution(generator);
            new_mean[d] += temp[d];
        }
        cur_vec_num += numItems;
        random_items.emplace_back(temp);
    }
    std::random_shuffle(random_items.begin(), random_items.end());
    for (int i = 0; i < dim; ++i) {
        new_mean[i] /= cur_vec_num;
    }
    // construct graph
    HNSWGraph my_hnswgraph(10, 30, 30, 10, 2);
    std::vector<LVQData> compress_items;
    for (int i = 0; i < numItems; i++) {
        if (i % 10000 == 0) std::cout << i << std::endl;

        CompressTo(random_items[i], compress_items[i], new_mean, dim);
        my_hnswgraph.Insert(compress_items[i]);
    }

    double total_brute_force_time = 0.0;
    double total_hnsw_time = 0.0;

    std::cout << "START QUERY" << std::endl;
    int numHits = 0;
    for (int i = 0; i < numQueries; i++) {
        // Generate random query
        std::vector<double> temp(dim);
        for (int d = 0; d < dim; d++) {
            temp[d] = distribution(generator);
        }
        //CompressTo(vec, query.inner_.get());
        //return query;
    }
        Point query(temp);

        // Brute force
        clock_t begin_time = clock();
        std::vector<std::pair<double, int>> distPairs;
        for (int j = 0; j < numItems; j++) {
            if (j == i) continue;
            distPairs.emplace_back(query.dist(random_items[j]), j);
        }
        std::sort(distPairs.begin(), distPairs.end());
        total_brute_force_time += double( clock () - begin_time ) /  CLOCKS_PER_SEC;

        begin_time = clock();
        std::vector<int> knns = my_hnswgraph.KNNSearch(query, K);
        total_hnsw_time += double( clock () - begin_time ) /  CLOCKS_PER_SEC;

        if (knns[0] == distPairs[0].second) numHits++;
    }
    std::cout << numHits << " " << total_brute_force_time / numQueries  << " " << total_hnsw_time / numQueries << std::endl;
}

int main() {
    RandomTest(10000, 4, 100, 5);
    return 0;
}
