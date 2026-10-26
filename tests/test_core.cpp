// SPDX-License-Identifier: Apache-2.0
#include "jacobsrays/traverse.hpp"
#include <cstdlib>
#include <iostream>
using namespace jacobsrays;
void require(bool value) { if (!value) std::abort(); }
int main() {
    Grid g={{3,2,1},{1,1,1},{0,0,0}}; g.validate();
    double s[]={.2,1.2,.5}, e[]={.8,1.2,.5};
    validate_ray(g,s,e);
    int count=0;
    walk(g,s,e,[&](index_t i,double l,double a,double b){
        require(i==3 && std::abs(l-.6)<1e-14 && a==0 && b==1); ++count;
    });
    require(count==1);
    double p[]={3,2,1},q[]={0,0,0};
    double sum=0; double previous=0;
    walk(g,p,q,[&](index_t i,double l,double a,double b){
        require(i>=0 && i<6 && l>0 && a==previous && b>a); previous=b; sum+=l;
    });
    require(std::abs(sum-std::sqrt(14.))<1e-13);
    bool rejected=false;
    try { Grid bad={{-1,2,1},{1,1,1},{0,0,0}}; bad.validate(); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected);
    std::cout<<"core checks passed\n";
}
