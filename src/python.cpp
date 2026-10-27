// SPDX-License-Identifier: Apache-2.0
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "jacobsrays/traverse.hpp"
#include <memory>
#include <new>

using namespace jacobsrays;
struct ArrayDeleter { void operator()(PyArrayObject* a) const { Py_XDECREF(a); } };
using Array = std::unique_ptr<PyArrayObject, ArrayDeleter>;
static Array array(PyObject* o, int type) {
    return Array(reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(o, type, NPY_ARRAY_IN_ARRAY)));
}
static Array empty(npy_intp n, int type) {
    return Array(reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &n, type)));
}
static bool vector3(const Array& a) {
    return a && PyArray_NDIM(a.get()) == 1 && PyArray_SIZE(a.get()) == 3;
}

static PyObject* run(PyObject* args, int mode, bool atomic_adjoint = false) {
    #ifndef _OPENMP
    if (atomic_adjoint) {
        PyErr_SetString(PyExc_RuntimeError,"atomic adjoint requires an OpenMP build (JACOBSRAYS_OPENMP=1)");
        return nullptr;
    }
    #endif
    PyObject *os, *oe, *on, *oh, *oo, *ox = nullptr;
    if (!PyArg_ParseTuple(args, mode ? "OOOOOO" : "OOOOO", &os, &oe, &on, &oh, &oo, &ox)) return nullptr;
    Array s=array(os, NPY_DOUBLE), e=array(oe, NPY_DOUBLE), n=array(on, NPY_INT64),
          h=array(oh, NPY_DOUBLE), o=array(oo, NPY_DOUBLE);
    if (!s || !e || !n || !h || !o) return nullptr;
    if (!vector3(n) || !vector3(h) || !vector3(o) ||
        PyArray_NDIM(s.get()) != 2 || PyArray_NDIM(e.get()) != 2 ||
        PyArray_DIM(s.get(),1) != 3 || PyArray_DIM(e.get(),1) != 3 ||
        PyArray_DIM(s.get(),0) != PyArray_DIM(e.get(),0)) {
        PyErr_SetString(PyExc_ValueError,"expected (R,3) rays and three-element grid vectors"); return nullptr;
    }
    Grid g;
    for (int a=0; a<3; ++a) {
        g.n[a] = static_cast<npy_int64*>(PyArray_DATA(n.get()))[a];
        g.spacing[a] = static_cast<double*>(PyArray_DATA(h.get()))[a];
        g.origin[a] = static_cast<double*>(PyArray_DATA(o.get()))[a];
    }
    const npy_intp nr=PyArray_DIM(s.get(),0);
    const auto* starts=static_cast<double*>(PyArray_DATA(s.get()));
    const auto* ends=static_cast<double*>(PyArray_DATA(e.get()));
    try {
        g.validate();
        for (npy_intp r=0; r<nr; ++r) validate_ray(g, starts+3*r, ends+3*r);
    } catch (const std::exception& ex) { PyErr_SetString(PyExc_ValueError,ex.what()); return nullptr; }
    if (mode) {
        Array x=array(ox, NPY_DOUBLE);
        if (!x) return nullptr;
        if (PyArray_NDIM(x.get()) != 1 || PyArray_SIZE(x.get()) != (mode==1 ? g.size() : nr)) {
            PyErr_SetString(PyExc_ValueError,"input vector has incorrect size"); return nullptr;
        }
        Array y=empty(mode==1 ? nr : g.size(), NPY_DOUBLE);
        if (!y) return nullptr;
        const auto* in=static_cast<double*>(PyArray_DATA(x.get()));
        auto* out=static_cast<double*>(PyArray_DATA(y.get()));
        Py_BEGIN_ALLOW_THREADS
        if (mode==1) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (npy_intp r=0; r<nr; ++r) {
                double sum=0;
                walk(g,starts+3*r,ends+3*r,[&](index_t i,double l,double,double){sum+=l*in[i];});
                out[r]=sum;
            }
        } else {
            std::fill(out,out+g.size(),0.0);
            #ifdef _OPENMP
            if (atomic_adjoint) {
                // One shared output volume. Every concurrent write is atomic;
                // the loop's implicit barrier completes updates before return.
                #pragma omp parallel for schedule(static)
                for (npy_intp r=0; r<nr; ++r)
                    walk(g,starts+3*r,ends+3*r,[&](index_t i,double l,double,double){
                        const double value=l*in[r];
                        #pragma omp atomic
                        out[i]+=value;
                    });
            } else
            #endif
            {
                // Fixed ray order remains the default, even in OpenMP builds.
                for (npy_intp r=0; r<nr; ++r)
                    walk(g,starts+3*r,ends+3*r,[&](index_t i,double l,double,double){out[i]+=l*in[r];});
            }
        }
        Py_END_ALLOW_THREADS
        return reinterpret_cast<PyObject*>(y.release());
    }
    if (nr == NPY_MAX_INTP) { PyErr_SetString(PyExc_OverflowError,"too many rays"); return nullptr; }
    Array offsets=empty(nr+1,NPY_INT64);
    if (!offsets) return nullptr;
    auto* ptr=static_cast<npy_int64*>(PyArray_DATA(offsets.get()));
    ptr[0]=0;
    Py_BEGIN_ALLOW_THREADS
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (npy_intp r=0; r<nr; ++r) {
        npy_int64 count=0;
        walk(g,starts+3*r,ends+3*r,[&](index_t,double,double,double){++count;});
        ptr[r+1]=count;
    }
    Py_END_ALLOW_THREADS
    for (npy_intp r=0; r<nr; ++r) {
        if (ptr[r+1] > NPY_MAX_INTP/8 - ptr[r]) {
            PyErr_SetString(PyExc_OverflowError,"intersection arrays are too large; use ray batches"); return nullptr;
        }
        ptr[r+1]+=ptr[r];
    }
    const npy_intp count=ptr[nr];
    Array indices=empty(count,NPY_INT64), lengths=empty(count,NPY_DOUBLE),
          enter=empty(count,NPY_DOUBLE), leave=empty(count,NPY_DOUBLE);
    if (!indices || !lengths || !enter || !leave) return nullptr;
    auto* ids=static_cast<npy_int64*>(PyArray_DATA(indices.get()));
    auto* lens=static_cast<double*>(PyArray_DATA(lengths.get()));
    auto* tin=static_cast<double*>(PyArray_DATA(enter.get()));
    auto* tout=static_cast<double*>(PyArray_DATA(leave.get()));
    Py_BEGIN_ALLOW_THREADS
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (npy_intp r=0; r<nr; ++r) {
        npy_int64 k=ptr[r];
        walk(g,starts+3*r,ends+3*r,[&](index_t i,double l,double a,double b){
            ids[k]=i; lens[k]=l; tin[k]=a; tout[k]=b; ++k;
        });
    }
    Py_END_ALLOW_THREADS
    return Py_BuildValue("NNNNN",offsets.release(),indices.release(),lengths.release(),enter.release(),leave.release());
}
static PyObject* trace(PyObject*,PyObject* args) { return run(args,0); }
static PyObject* project(PyObject*,PyObject* args) { return run(args,1); }
static PyObject* backproject(PyObject*,PyObject* args) { return run(args,2); }
static PyObject* backproject_atomic(PyObject*,PyObject* args) { return run(args,2,true); }
static PyMethodDef methods[] = {
    {"trace",trace,METH_VARARGS,"Ordered batched voxel intersections."},
    {"project",project,METH_VARARGS,"Matrix-free scalar forward operation."},
    {"backproject",backproject,METH_VARARGS,"Matrix-free Euclidean adjoint."},
    {"backproject_atomic",backproject_atomic,METH_VARARGS,"OpenMP atomic Euclidean adjoint; summation order may vary."},
    {nullptr,nullptr,0,nullptr}
};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_native",nullptr,-1,methods};
PyMODINIT_FUNC PyInit__native() {
    import_array();
    PyObject* m=PyModule_Create(&module);
    if (!m) return nullptr;
    #ifdef _OPENMP
    const int openmp_enabled=1;
    #else
    const int openmp_enabled=0;
    #endif
    if (PyModule_AddIntConstant(m,"openmp_enabled",openmp_enabled)<0) {
        Py_DECREF(m); return nullptr;
    }
    return m;
}
