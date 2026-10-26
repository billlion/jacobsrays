"""Repeatable warm-cache throughput/memory measurements; no speedup claim."""
import argparse
import json
import os
import platform
import time
import numpy as np
from jacobsrays import Grid, trace, RayTransform


def measure(function, repeat=5):
    function()
    times=[]
    for _ in range(repeat):
        t=time.perf_counter(); value=function(); times.append(time.perf_counter()-t)
    return min(times), value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rays',type=int,default=20000)
    parser.add_argument('--size',type=int,default=64)
    args=parser.parse_args()
    rng=np.random.default_rng(2026)
    grid=Grid((args.size,)*3)
    starts=rng.uniform(0,args.size,(args.rays,3)); starts[:,0]=-1
    ends=rng.uniform(0,args.size,(args.rays,3)); ends[:,0]=args.size+1
    op=RayTransform(grid,starts,ends)
    image=rng.normal(size=grid.image_shape); data=rng.normal(size=args.rays)
    tracing,paths=measure(lambda: trace(grid,starts,ends))
    forward,_=measure(lambda: op.direct(image))
    adjoint,_=measure(lambda: op.adjoint(data))
    result=dict(python=platform.python_version(),platform=platform.platform(),
                numpy=np.__version__,omp_num_threads=os.getenv('OMP_NUM_THREADS'),
                rays=args.rays,grid=grid.shape,intersections=len(paths.indices),
                trace_seconds=tracing,forward_seconds=forward,adjoint_seconds=adjoint,
                trace_rays_per_second=args.rays/tracing,
                path_bytes=sum(a.nbytes for a in (paths.offsets,paths.indices,paths.lengths,paths.t_enter,paths.t_exit)))
    try:
        csr=paths.to_csr()
        result['cached_csr_forward_seconds']=measure(lambda: csr@image.ravel())[0]
        result['cached_csr_adjoint_seconds']=measure(lambda: csr.T@data)[0]
        result['csr_bytes']=sum(a.nbytes for a in (csr.data,csr.indices,csr.indptr))
    except ImportError:
        pass
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
