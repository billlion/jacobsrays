# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from jacobsrays import Grid, trace, RayTransform, TensorRayTransform


def reference(grid, start, end):
    """Independent oracle: intersect every plane, sort parameters, use midpoints."""
    start, end = np.asarray(start), np.asarray(end)
    d = end-start
    cuts = [0., 1.]
    for a, n in enumerate(grid.shape):
        if d[a] != 0:
            cuts.extend((grid.origin[a]+np.arange(n+1)*grid.spacing[a]-start[a])/d[a])
    cuts = np.unique(np.clip(cuts, 0, 1))
    ids, lengths = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        pos = start+((lo+hi)/2)*d
        cell = np.floor((pos-grid.origin)/grid.spacing).astype(int)
        if np.all(cell >= 0) and np.all(cell < grid.shape) and hi > lo:
            ids.append(np.ravel_multi_index(tuple(cell), grid.shape, order='F'))
            lengths.append((hi-lo)*np.linalg.norm(d))
    return np.array(ids, dtype=np.int64), np.array(lengths)


class TraversalTests(unittest.TestCase):
    def test_axis_partial_rectangular(self):
        g = Grid((3, 2))
        p = trace(g, [[-1,.5], [.2,1.2]], [[4,.5], [.8,1.2]])
        assert_array_equal(p.offsets, [0,3,4])
        assert_array_equal(p.indices, [0,1,2,3])
        assert_allclose(p.lengths, [1,1,1,.6])
        assert_allclose(p.t_enter, [.2,.4,.6,0])
        assert_allclose(p.t_exit, [.4,.6,.8,1])

    def test_boundary_tie_and_reversal(self):
        g = Grid((3,3,3))
        p = trace(g, [-1,-1,-1], [4,4,4])
        q = trace(g, [4,4,4], [-1,-1,-1])
        assert_array_equal(p.indices, [0,13,26])
        assert_array_equal(q.indices, p.indices[::-1])
        assert_allclose(p.lengths, np.sqrt(3))
        assert_allclose(q.t_enter, 1-p.t_exit[::-1])
        for coordinate, expected in [(0, [0,1,2]), (1, [3,4,5]), (3, [])]:
            p = trace(Grid((3,3)), [-1,coordinate], [4,coordinate])
            assert_array_equal(p.indices, expected)

    def test_empty_miss_tangent_degenerate(self):
        g = Grid((2,2))
        p = trace(g, np.empty((0,2)), np.empty((0,2)))
        assert_array_equal(p.offsets, [0])
        self.assertEqual(p.to_csr().shape, (0,4))
        p = trace(g, [[3,3], [0,0], [-1,1]], [[4,4], [0,0], [1,-1]])
        assert_array_equal(p.offsets, [0,0,0,0])
        self.assertEqual(p.to_csr().shape, (3,4))
        self.assertEqual(p.to_csr().nnz, 0)

    def test_tiny_nonzero_and_negative(self):
        # Legacy fixed absolute 1e-8 direction tolerance discards this ray.
        p = trace(Grid((3,2), (1e-10,2e-10), (-1e-10,-2e-10)),
                  [-2e-10,-1e-10], [3e-10,-1e-10])
        assert_array_equal(p.indices, [0,1,2])
        assert_allclose(p.lengths, 1e-10, atol=1e-24)
        p = trace(Grid((2,2)), [1,1], [.25,.25])
        assert_array_equal(p.indices, [0])
        assert_allclose(p.lengths, [.75*np.sqrt(2)])

    def test_random_oracle_and_reversal(self):
        rng = np.random.default_rng(921)
        for g in [Grid((5,7), (.3,1.7), (-2,3)), Grid((4,7,3), (.2,1.1,2.3), (4,-2,8))]:
            low = np.array(g.origin)-np.array(g.spacing)*2
            high = np.array(g.origin)+(np.array(g.shape)+2)*g.spacing
            s, e = rng.uniform(low,high,(2,2000,g.ndim))
            p, rev = trace(g,s,e), trace(g,e,s)
            for r in range(len(s)):
                k, j = p.ray(r), rev.ray(r)
                ids, lens = reference(g,s[r],e[r])
                assert_array_equal(p.indices[k], ids)
                assert_allclose(p.lengths[k], lens, atol=3e-13, rtol=1e-10)
                assert_array_equal(rev.indices[j], p.indices[k][::-1])
                assert_allclose(rev.lengths[j], p.lengths[k][::-1], atol=3e-13)
                self.assertTrue(np.all(p.lengths[k] > 0))
                assert_allclose(p.t_exit[k][:-1], p.t_enter[k][1:], atol=0, rtol=0)

    def test_near_corner_and_strided_input(self):
        g = Grid((7,9,5))
        s = np.array([[-1.,-1.,-1.], [10,10,10], [1,2,3]])
        e = np.array([[10,10+1e-10,10-1e-9], [-1,-1,-1], [5,6,4]])
        for starts, ends in [(s,e),(s[::-1],e[::-1])]:
            p = trace(g,starts,ends)
            for r in range(3):
                ids,lens=reference(g,starts[r],ends[r])
                # Compare voxel sums, allowing sub-ulp intervals at exact ties.
                got=np.bincount(p.indices[p.ray(r)],p.lengths[p.ray(r)],minlength=315)
                want=np.bincount(ids,lens,minlength=315)
                assert_allclose(got,want,atol=1e-13)

    def test_invalid(self):
        for args in [((0,2),), ((2.5,2),), ((2,2),(0,1)), ((2,2),None,(np.nan,0)), ((2**52,2**52),)]:
            with self.assertRaises(ValueError): Grid(*args)
        g=Grid((2,2))
        for s,e in [([0,0],[1,1,1]),([0,np.nan],[1,1]),([0j,0],[1,1]),([1e308,0],[-1e308,0])]:
            with self.assertRaises(ValueError): trace(g,s,e)
        with self.assertRaises(IndexError): trace(g,[0,0],[1,1]).ray(-1)
        with self.assertRaises(ValueError): RayTransform(g,[0,0],[1,1]).direct(np.ones((2,2),complex))

    def test_scalar_adjoint_and_sparse(self):
        rng=np.random.default_rng(32)
        g=Grid((5,7,3),(.2,.3,.4),(-1,-2,3))
        s,e=rng.uniform([-2,-3,2],[2,2,6],(2,137,3))
        op=RayTransform(g,s,e); A=op.paths().to_csr()
        x=rng.normal(size=g.image_shape); y=rng.normal(size=137)
        assert_allclose(op.direct(x),A@x.ravel(),atol=1e-14)
        assert_allclose(op.adjoint(y).ravel(),A.T@y,atol=1e-14)
        assert_allclose(np.vdot(op.direct(x),y),np.vdot(x,op.adjoint(y)),atol=1e-13)
        assert_allclose(op.as_scipy()@x.ravel(),op.direct(x))
        # Endpoint arrays are owned by the operator.
        s[:]=100
        assert_allclose(op.direct(x),A@x.ravel(),atol=1e-14)

    def test_final_matrix_entry(self):
        p=trace(Grid((2,2)),[0,1.5],[2,1.5])
        self.assertEqual(p.to_csr()[0,3],1)

    def test_tensor_adjoint_and_rank(self):
        rng=np.random.default_rng(91)
        for dim in [2,3]:
            g=Grid((3,)*dim)
            s,e=rng.uniform(-1,4,(2,39,dim))
            for rank in [0,1,2,3]:
                op=TensorRayTransform(g,s,e,rank)
                x=rng.normal(size=op.domain_shape); y=rng.normal(size=39)
                assert_allclose(np.vdot(op.direct(x),y),np.vdot(x,op.adjoint(y)),atol=1e-12)
                backward=TensorRayTransform(g,e,s,rank)
                assert_allclose(backward.direct(x),(-1)**rank*op.direct(x),atol=1e-12)
                assert_allclose(op.as_scipy()@x.ravel(),op.direct(x))
            op=TensorRayTransform(g,s,e,2)
            identity=np.zeros(op.domain_shape)
            for j,alpha in enumerate(op.exponents):
                if max(alpha)==2: identity[j]=1
            assert_allclose(op.direct(identity),op.scalar.direct(np.ones(g.image_shape)),atol=1e-13)


if __name__ == '__main__': unittest.main()
