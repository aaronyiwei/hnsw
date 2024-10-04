#include "hnsw.h"

#include <algorithm>
#include <cassert>
#include <ctime>
#include <iostream>
#include <utility>
#include <cstring>
#include <random>
#include <vector>
#include <math.h>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
size_t max_d,max_n;

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}
// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);

}
double elapsed(){
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (tv.tv_sec + tv.tv_usec * 1e-6);
}
std::pair<float, float> MakeLocalCache(const int8_t *c, float scale, int dim) {
        int64_t norm1 = 0;
        int64_t norm2 = 0;
        for (int i = 0; i < dim; ++i) {
            norm1 += c[i];
            norm2 += c[i] * c[i];
        }
        return {norm1 * scale, norm2 * scale * scale};
    }
 void CompressTo(const float* src, LVQData& dest, const float *mean_, int dim) {
        if (true) {
            float norm = 0;
            float *src_without_const = const_cast<float *>(src);
            for (int j = 0; j < dim; ++j) {
                norm += src_without_const[j] * src_without_const[j];
            }
            norm = std::sqrt(norm);
            if (norm == 0) {
                std::fill(dest.compress_vec_, dest.compress_vec_ + dim, 0);
            } else {
                for (int j = 0; j < dim; ++j) {
                    src_without_const[j] /= norm;
                }
            }
        }
        //int8_t *compress = dest.compress_vec_;

        float lower = std::numeric_limits<float>::max();
        float upper = -std::numeric_limits<float>::max();
        for (int j = 0; j < dim; ++j) {
            auto x = static_cast<float>(src[j]-mean_[j]);
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
        dest.scale_ = scale;
        dest.bias_ = bias;
        dest.local_cache_ = MakeLocalCache(dest.compress_vec_, scale, dim);
}

int main() {
    //RandomTest(10000, 4, 100, 5);
    int dim = 128;               // Dimension of the elements
    int max_elements = 1000000;   
    int M = 128 ;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    float* x;
    // Initing index
    float mean[dim+1];
    memset(mean,0.0,sizeof(mean));
    int cur_vec_num = 0;
    x=fvecs_read("/home/infominer/aaron/dataset/siftsmall/siftsmall_base.fvecs",&max_d,&max_n);
    //x=fvecs_read("/Users/aaron/Downloads/siftsmall/siftsmall_base.fvecs",&max_d,&max_n);
    for(int i = 0; i < max_n; i++){
        float * xv = x + i * max_d;
        for (int d = 0; d < max_d; d++) {
            mean[d] += xv[d];
        }
        cur_vec_num += max_d;
    }
    for(int i = 0; i < dim; ++i) {
        mean[i] =0; // /= max_n;
    }
    std::cout<<std::endl;
    std::vector<LVQData> compress_items(max_n);
    HNSWGraph my_hnswgraph(10, 30, 30, 100, 5);
    for (int i = 0; i < max_n; i++) {
        if (i % 1000 == 0) std::cout << i << std::endl;
        CompressTo(x + i * max_d, compress_items[i], mean, dim);
        my_hnswgraph.Insert(compress_items[i], mean);
    }
    /*
    std::string hnsw_path = "hnsw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;*/

    delete[] x;
    //delete compress_items;


    size_t nq,d2;
    float* xq;
    double t0 = elapsed();
    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        xq = fvecs_read("/home/infominer/aaron/dataset/siftsmall/siftsmall_query.fvecs", &d2, &nq);
        //xq = fvecs_read("/Users/aaron/Downloads/siftsmall/siftsmall_query.fvecs", &d2, &nq);
    }
    size_t k,nq2;         // nb of results per query in the GT
    int* gt; // nq * k matrix of ground-truth nearest-neighbors
    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        gt = ivecs_read("/home/infominer/aaron/dataset/siftsmall/siftsmall_groundtruth.ivecs", &k, &nq2);
        //gt = ivecs_read("/Users/aaron/Downloads/siftsmall/siftsmall_groundtruth.ivecs", &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");
    }
    { // Use the found configuration to perform a search
        printf("[%.3f s] Perform a search on %ld queries\n",
               elapsed() - t0,
               nq);
        std::cout<<d2<<std::endl;
        // output buffers
        double t1=elapsed();
        int correct=0,cnt=0;
        int numHits = 0, cur_quevec_num=0;
        std::vector<LVQData> compress_queries(nq+1);
        std::cout<<nq<<std::endl;
        for (int i = 0; i < nq; i++) {
            CompressTo(xq + i * d2, compress_queries[i], mean, d2);
            std::vector<int> knns = my_hnswgraph.KNNSearch(compress_queries[i], 100, mean);
            //std::vector<int> knns = my_hnswgraph.KNNSearch(xq + i * d2, 100, mean, query_mean);
            std::cout<<knns[0]<<' '<<gt[cnt]<<std::endl;
            if (knns[0] == gt[cnt]) numHits++;
            cnt+=k;
        }
        printf("[%.3f s] Compute recalls\n", elapsed() - t1);
        //for(int i=0; i<k; i++) std::cout<<gt[i]<<std::endl;
        std::cout<<nq*k<<' '<<d2<<std::endl;
        printf("%.4f\n", numHits / float(nq));
    }
    delete[] xq;
    delete[] gt;
    return 0;
}
