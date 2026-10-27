# SPDX-License-Identifier: Apache-2.0
"""Optional ABI/ownership checks for the experiment; requires oneMKL.

Run with GNU/LP64 configured before Python starts, as in mkl.yml.
These tests are deliberately outside the normal dependency-free test suite.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from mkl_probe import MKL, MKLMatrix


class MKLProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = MKL(Path(sys.prefix)/"lib/libmkl_rt.so.3")

    def test_nonsquare_unsorted_duplicates_and_owned_storage(self):
        # Empty row, repeated column, unsorted column indices: an independent
        # dense oracle also checks which dimension is the leading dimension.
        a = csr_matrix(([3., 2., -1., 7., 5.], [3, 1, 3, 4, 0], [0, 3, 3, 5]),
                       shape=(3, 5))
        expected = np.array([[0., 2., 0., 2., 0.], [0., 0., 0., 0., 0.],
                             [5., 0., 0., 0., 7.]])
        with MKLMatrix(a, self.runtime) as matrix:
            a.data[:] = 99  # The foreign handle must not borrow caller storage.
            for transpose in (False, True):
                dense = expected.T if transpose else expected
                for columns in (1, 3, 6):
                    x = np.arange(dense.shape[1]*columns, dtype=float).reshape(-1, columns)/7
                    padded = np.empty((x.shape[0], columns*2)); padded[:, ::2] = x
                    for values in (x, np.asfortranarray(x), padded[:, ::2]):
                        assert_allclose(matrix.dot(values, transpose), dense@values, atol=1e-13)
                    assert_allclose(matrix.dot(x[:, 0], transpose), dense@x[:, 0], atol=1e-13)
        matrix.close()  # Idempotent; no double-free.
        with self.assertRaises(RuntimeError):
            matrix@np.ones(5)

    def test_zero_rays_missed_rays_and_zero_components(self):
        for shape in ((0, 5), (3, 5), (3, 0), (0, 0)):
            with MKLMatrix(csr_matrix(shape), self.runtime) as matrix:
                for transpose in (False, True):
                    rows, cols = shape[::-1] if transpose else shape
                    for tail in ((), (0,), (6,)):
                        actual = matrix.dot(np.ones((cols, *tail)), transpose)
                        assert_array_equal(actual, np.zeros((rows, *tail)))
        with MKLMatrix(csr_matrix(np.eye(3)), self.runtime) as matrix:
            assert_array_equal(matrix@np.empty((3, 0)), np.empty((3, 0)))

    def test_invalid_inputs_fail_before_calling_foreign_code(self):
        with self.assertRaises(ValueError):
            MKLMatrix(csr_matrix([[1j]]), self.runtime)
        with MKLMatrix(csr_matrix(np.eye(3)), self.runtime) as matrix:
            for values in (np.ones(2), np.ones((3, 2, 1)), np.ones(3, dtype=complex)):
                with self.assertRaises(ValueError):
                    matrix@values


if __name__ == "__main__":
    unittest.main()
