// SPDX-License-Identifier: Apache-2.0
#include "mex.h"
#include "jacobsrays/traverse.hpp"
#include <limits>
#include <vector>
using namespace jacobsrays;

static bool real_double(const mxArray* x) {
    return mxIsDouble(x) && !mxIsComplex(x) && !mxIsSparse(x) && mxGetNumberOfDimensions(x)==2;
}
static void fail(const char* message) { mexErrMsgIdAndTxt("jacobsrays:invalidInput", "%s", message); }

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs!=5 || nlhs!=5) fail("Use jacobsrays(shape, starts, ends, spacing, origin).");
    for (int a=0; a<5; ++a) if (!real_double(prhs[a])) fail("Inputs must be full real double arrays.");
    const mwSize dim=mxGetNumberOfElements(prhs[0]), nr=mxGetM(prhs[1]);
    if ((dim!=2 && dim!=3) || mxGetNumberOfElements(prhs[3])!=dim || mxGetNumberOfElements(prhs[4])!=dim ||
        mxGetN(prhs[1])!=dim || mxGetM(prhs[2])!=nr || mxGetN(prhs[2])!=dim)
        fail("Expected 2D/3D grid vectors and matching R-by-dimension endpoint arrays.");
    const double *n=mxGetPr(prhs[0]), *s=mxGetPr(prhs[1]), *e=mxGetPr(prhs[2]),
                 *h=mxGetPr(prhs[3]), *o=mxGetPr(prhs[4]);
    Grid g={{1,1,1},{1,1,1},{0,0,0}};
    for (mwSize a=0; a<dim; ++a) {
        if (!std::isfinite(n[a]) || n[a]<1 || n[a]>4503599627370496. || n[a]!=std::floor(n[a]))
            fail("Grid dimensions must be positive integers no larger than 2^52.");
        g.n[a]=static_cast<index_t>(n[a]); g.spacing[a]=h[a]; g.origin[a]=o[a];
    }
    auto endpoints=[&](mwSize r, double* p, double* q) {
        p[2]=q[2]=.5;
        for (mwSize a=0; a<dim; ++a) { p[a]=s[r+nr*a]; q[a]=e[r+nr*a]; }
    };
    try {
        g.validate();
        for (mwSize r=0; r<nr; ++r) { double p[3],q[3]; endpoints(r,p,q); validate_ray(g,p,q); }
    } catch (const std::exception& ex) { fail(ex.what()); }
    if (nr==std::numeric_limits<mwSize>::max()) fail("Too many rays.");
    plhs[0]=mxCreateNumericMatrix(nr+1,1,mxINT64_CLASS,mxREAL);
    auto* ptr=static_cast<int64_T*>(mxGetData(plhs[0]));
    ptr[0]=1;  // MATLAB offsets are one-based, just like voxel indices.
    for (mwSize r=0; r<nr; ++r) {
        double p[3],q[3]; endpoints(r,p,q); index_t count=0;
        walk(g,p,q,[&](index_t,double,double,double){++count;});
        if (count>std::numeric_limits<index_t>::max()/8-ptr[r]) fail("Too many intersections; use ray batches.");
        ptr[r+1]=ptr[r]+count;
    }
    const mwSize count=static_cast<mwSize>(ptr[nr]-1);
    plhs[1]=mxCreateNumericMatrix(count,1,mxINT64_CLASS,mxREAL);
    for (int a=2; a<5; ++a) plhs[a]=mxCreateDoubleMatrix(count,1,mxREAL);
    auto* ids=static_cast<int64_T*>(mxGetData(plhs[1]));
    double *lens=mxGetPr(plhs[2]), *tin=mxGetPr(plhs[3]), *tout=mxGetPr(plhs[4]);
    for (mwSize r=0; r<nr; ++r) {
        double p[3],q[3]; endpoints(r,p,q); index_t k=ptr[r]-1;
        walk(g,p,q,[&](index_t i,double l,double a,double b) {
            ids[k]=i+1; lens[k]=l; tin[k]=a; tout[k]=b; ++k;
        });
    }
}
