# SPDX-License-Identifier: Apache-2.0
"""Slice-by-slice reconstruction of the published Desai--Lionheart f33 phantom.

Run: python examples/trt_slices.py
     python examples/trt_slices.py --sizes 16 --views 30 --detectors 49 --maxiter 100
     python examples/trt_slices.py --json

Only the f33 component of Table 1 is used; the other tensor components are
zero. This is a diagonal-component example, not the paper's full three-axis
algorithm. See docs/trt_slices.md for attribution, conventions and limits.
"""
import argparse
import json

import numpy as np
from scipy.special import erf
from scipy.sparse.linalg import lsmr

from jacobsrays import Grid, RayTransform, TransverseRayTransform


BETA = 50.
SLICES = (-.5, .5)
# Table 1: (amplitude, centre), physical f33 (no off-diagonal packing factor).
F33_TERMS = ((1., (.5, -.5, -.5)),
             (-1., (.5, -.5, .5)),
             (1., (.5, .5, .5)))


def phantom_f33(x, y, z):
    """Published Gaussian sum, set to zero outside [-1,1)^3."""
    value = sum(alpha*np.exp(-BETA*((x-a[0])**2+(y-a[1])**2+(z-a[2])**2))
                for alpha, a in F33_TERMS)
    inside = (x >= -1) & (x < 1) & (y >= -1) & (y < 1) & (z >= -1) & (z < 1)
    return np.where(inside, value, 0.)


def planar_rays(views, detectors):
    """Parallel rays in xy; a small fixed angle offset avoids grid-plane ties."""
    angles = np.arange(views)*np.pi/views+.013
    u = np.column_stack((np.cos(angles), np.sin(angles)))
    normal = np.column_stack((-u[:, 1], u[:, 0]))
    offsets = np.linspace(-1.45, 1.45, detectors)
    p = (normal[:, None, :]*offsets[None, :, None]).reshape(-1, 2)
    direction = np.repeat(u, detectors, axis=0)
    return p-3*direction, p+3*direction


def analytic_f33(starts, ends, z):
    """Exact finite-segment Gaussian integrals at height z, clipped to the cube.

    This reference uses slab clipping and error functions, with no voxel
    tracing, interpolation or reconstruction matrix. Boundary-parallel rays
    follow the same half-open cube convention as the voxel field.
    """
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    d = ends-starts
    length = np.hypot(d[:, 0], d[:, 1])
    u = np.divide(d, length[:, None], out=np.zeros_like(d), where=length[:, None] != 0)
    low, high = np.zeros(len(d)), length.copy()
    hit = (length > 0) & (-1 <= z) & (z < 1)
    for axis in range(2):
        moving = u[:, axis] != 0
        t0 = np.divide(-1-starts[:, axis], u[:, axis],
                       out=np.full(len(d), -np.inf), where=moving)
        t1 = np.divide(1-starts[:, axis], u[:, axis],
                       out=np.full(len(d), np.inf), where=moving)
        low = np.maximum(low, np.minimum(t0, t1))
        high = np.minimum(high, np.maximum(t0, t1))
        hit &= moving | ((starts[:, axis] >= -1) & (starts[:, axis] < 1))
    hit &= high > low
    result = np.zeros(len(d))
    for alpha, centre in F33_TERMS:
        delta = starts-np.array(centre[:2])
        mu = -np.sum(delta*u, axis=1)
        perpendicular = delta+mu[:, None]*u
        rho2 = np.sum(perpendicular**2, axis=1)+(z-centre[2])**2
        result += alpha*np.exp(-BETA*rho2)*np.sqrt(np.pi)/(2*np.sqrt(BETA))* (
            erf(np.sqrt(BETA)*(high-mu))-erf(np.sqrt(BETA)*(low-mu)))
    return np.where(hit, result, 0.)


def relative_error(actual, reference):
    return float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))


def solve_slice(matrix, data, truth, maxiter):
    """Unregularised LSMR; expose stopping diagnostics rather than hide them."""
    result = lsmr(matrix, data, atol=1e-10, btol=1e-10, maxiter=maxiter)
    image, stop, iterations = result[:3]
    diagnostics = {
        'relative_image_error': relative_error(image, truth.ravel()),
        'relative_data_residual': relative_error(matrix@image, data),
        'stop_code': int(stop),
        'iterations': int(iterations),
    }
    return image, diagnostics


def run_case(n, views=90, detectors=129, maxiter=300):
    """Compare TRT, scalar-voxel, and analytic data on two voxel-centre planes."""
    h = 2/n
    grid = Grid((n, n, 2), (h, h, 1.), (-1, -1, -1))
    coordinates = -1+(np.arange(n)+.5)*h
    z, y, x = np.meshgrid(SLICES, coordinates, coordinates, indexing='ij')
    truth = phantom_f33(x, y, z)
    tensor = np.zeros((6, 2, n, n))
    tensor[5] = truth  # Packed order: xx, sqrt(2)xy, sqrt(2)xz, yy, sqrt(2)yz, zz.
    starts2, ends2 = planar_rays(views, detectors)
    starts = np.concatenate([np.column_stack((starts2, np.full(len(starts2), z))) for z in SLICES])
    ends = np.concatenate([np.column_stack((ends2, np.full(len(ends2), z))) for z in SLICES])
    trt = TransverseRayTransform(grid, starts, ends)
    data = trt.direct(tensor)[:, 5].reshape(2, -1)
    scalar = RayTransform(Grid((n, n), (h, h), (-1, -1)), starts2, ends2)
    matrix = scalar.paths().to_csr()
    baseline = np.stack([matrix@f.ravel() for f in truth])
    analytic = np.stack([analytic_f33(starts2, ends2, z) for z in SLICES])
    reconstructions = []
    for k, z in enumerate(SLICES):
        xt, trt_info = solve_slice(matrix, data[k], truth[k], maxiter)
        xs, scalar_info = solve_slice(matrix, baseline[k], truth[k], maxiter)
        _, analytic_info = solve_slice(matrix, analytic[k], truth[k], maxiter)
        reconstructions.append({
            'z': z, 'trt_data': trt_info, 'scalar_data': scalar_info,
            'analytic_data': analytic_info,
            'trt_vs_scalar_reconstruction_relative': relative_error(xt, xs),
        })
    return {
        'n': n, 'views': views, 'detectors': detectors, 'maxiter': maxiter,
        'trt_vs_scalar_projection_relative': relative_error(data, baseline),
        'voxel_vs_analytic_projection_relative': relative_error(data, analytic),
        'reconstructions': reconstructions,
    }


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=positive_integer, default=[32, 64])
    parser.add_argument('--views', type=positive_integer, default=90)
    parser.add_argument('--detectors', type=positive_integer, default=129)
    parser.add_argument('--maxiter', type=positive_integer, default=300)
    parser.add_argument('--json', action='store_true', help='print machine-readable results')
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.views < 2 or args.detectors < 3:
        parser.error('sizes/views must be at least 2, detectors at least 3')
    results = [run_case(n, args.views, args.detectors, args.maxiter) for n in args.sizes]
    if args.json:
        print(json.dumps(results, indent=2, allow_nan=False))
        return
    print('Published f33, two slices at z=-0.5,+0.5; other tensor components zero.')
    print('All reconstructions use the scalar 2D matrix and unregularised LSMR.')
    for result in results:
        print(f"\n{result['n']}x{result['n']}, {result['views']} views, {result['detectors']} detectors")
        print(f"  TRT / scalar projection difference: {result['trt_vs_scalar_projection_relative']:.3g}")
        print(f"  Voxel / analytic projection error: {result['voxel_vs_analytic_projection_relative']:.3%}")
        for entry in result['reconstructions']:
            a, v = entry['analytic_data'], entry['trt_data']
            print(f"  z={entry['z']:+g}: voxel-data image error={v['relative_image_error']:.3g}; "
                  f"analytic-data image error={a['relative_image_error']:.3%}; "
                  f"analytic solve: stop={a['stop_code']}, iterations={a['iterations']}")
    print('\nAnalytic-data reconstruction errors need not decrease with voxel refinement.')
    print('LSMR stop=7 means the iteration limit was reached; use --json for all diagnostics.')


if __name__ == '__main__':
    main()
