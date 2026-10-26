// SPDX-License-Identifier: Apache-2.0
// Benchmark harness only. The legacy source is supplied separately by the user.
extern "C" {
#include "jacobs_rays.h"
void mod_jacobs_ray_3d(int, double*, double*, int*, double*, int*, jacobs_options*);
}
#include "jacobsrays/traverse.hpp"
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
int main() {
    const int nr=200000, n=64;
    std::mt19937_64 rng(2026); std::uniform_real_distribution<double> unif(.001,n-.001);
    std::vector<std::array<double,3>> starts(nr),ends(nr);
    for (int r=0;r<nr;++r) { starts[r]={{-1,unif(rng),unif(rng)}}; ends[r]={{n+1.,unif(rng),unif(rng)}}; }
    jacobs_options options={0,0,0,n,n,n,0,0,0,1,1,1};
    jacobsrays::Grid g={{n,n,n},{1,1,1},{0,0,0}};
    double best[2]={1e20,1e20},sums[2]={};
    std::int64_t counts[2]={};
    for(int repeat=0;repeat<6;++repeat) for(int mode=0;mode<2;++mode) {
        double checksum=0; std::int64_t count=0;
        auto start=std::chrono::steady_clock::now();
        for(int r=0;r<nr;++r) {
            int ids[3*n],used=0; double lengths[3*n];
            if(mode==0) mod_jacobs_ray_3d(n,starts[r].data(),ends[r].data(),ids,lengths,&used,&options);
            else {
                jacobsrays::validate_ray(g,starts[r].data(),ends[r].data());
                jacobsrays::walk(g,starts[r].data(),ends[r].data(),[&](jacobsrays::index_t i,double l,double,double){ids[used]=int(i);lengths[used++]=l;});
            }
            for(int k=0;k<used;++k) checksum+=lengths[k]*(1.+ids[k]*1e-6);
            count+=used;
        }
        const double time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(repeat) best[mode]=std::min(best[mode],time);
        sums[mode]=checksum; counts[mode]=count;
    }
    std::cout<<std::setprecision(12)<<"{\"rays\":"<<nr<<",\"legacy_seconds\":"<<best[0]<<",\"new_seconds\":"<<best[1]
             <<",\"legacy_segments\":"<<counts[0]<<",\"new_segments\":"<<counts[1]
             <<",\"legacy_checksum\":"<<sums[0]<<",\"new_checksum\":"<<sums[1]<<"}\n";
    return counts[0]!=counts[1] || std::abs(sums[0]-sums[1])>1e-6*std::abs(sums[0]);
}
