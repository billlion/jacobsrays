# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the published-component slice reconstruction example."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

# Import the example by path: examples are source-distribution files, not API.
spec = importlib.util.spec_from_file_location(
    'trt_slices_example', Path(__file__).resolve().parents[1]/'examples'/'trt_slices.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class SliceTRTTests(unittest.TestCase):
    def test_analytic_integrals_against_quadrature(self):
        starts = np.array([[-2., -.5], [-2., -1.], [.1, .2], [2., 2.], [.4, -.5],
                           [.5, -2.], [-2., 1.], [-2., -1.]])
        ends = np.array([[2., -.5], [2., 1.], [.7, -.1], [3., 2.], [.4, -.5],
                         [.5, 2.], [2., 1.], [2., -1.]])
        for z in (-.5, .5, 1.1):
            reference = []
            for s, e in zip(starts, ends):
                d = e-s
                # Break quadrature at cube crossings to handle truncated tails.
                cuts = []
                for axis in range(2):
                    if d[axis] != 0:
                        cuts.extend(t for t in ((-1-s[axis])/d[axis], (1-s[axis])/d[axis]) if 0 < t < 1)
                value, _ = quad(lambda t: float(example.phantom_f33(*(s+t*d), z)),
                                0, 1, points=sorted(set(cuts)), epsabs=1e-12, epsrel=1e-11)
                reference.append(value*np.linalg.norm(d))
            assert_allclose(example.analytic_f33(starts, ends, z), reference, atol=2e-12, rtol=1e-10)
            assert_allclose(example.analytic_f33(ends, starts, z), reference, atol=2e-12, rtol=1e-10)

    def test_slice_projection_and_reconstruction(self):
        result = example.run_case(12, views=30, detectors=49, maxiter=150)
        self.assertLess(result['trt_vs_scalar_projection_relative'], 1e-12)
        self.assertTrue(np.isfinite(result['voxel_vs_analytic_projection_relative']))
        for entry in result['reconstructions']:
            self.assertLess(entry['trt_vs_scalar_reconstruction_relative'], 1e-9)
            self.assertLess(entry['trt_data']['relative_image_error'], 1e-5)
            self.assertLess(entry['trt_data']['relative_data_residual'], 1e-8)
            self.assertNotEqual(entry['trt_data']['stop_code'], 7)
            # No convergence-to-truth claim for the model-mismatched analytic
            # data: report its errors/iterations in the example, do not fit a
            # regression threshold to this particular reconstruction.
            self.assertTrue(np.isfinite(entry['analytic_data']['relative_image_error']))


if __name__ == '__main__':
    unittest.main()
