"""Real CIL integration checks; skip only when CIL is not installed."""
import importlib.util
import unittest
import numpy as np
from numpy.testing import assert_allclose


@unittest.skipUnless(importlib.util.find_spec('cil'), 'CIL is not installed')
class CILTests(unittest.TestCase):
    def test_image_geometry_and_out(self):
        from cil.framework import ImageGeometry
        from jacobsrays.cil import JacobProjector
        rng=np.random.default_rng(4)
        for dim in [2,3]:
            ig=ImageGeometry(voxel_num_x=4,voxel_num_y=3,
                             voxel_num_z=2 if dim==3 else 0,
                             voxel_size_x=.7,voxel_size_y=1.1,voxel_size_z=1.3,
                             center_x=2,center_y=-1,center_z=3,dtype=np.float64)
            s,e=rng.uniform(-5,7,(2,25,dim))
            op=JacobProjector(ig,s,e)
            x=ig.allocate(dtype=np.float64); x.fill(rng.normal(size=ig.shape))
            y=op.range_geometry().allocate(dtype=np.float64); y.fill(rng.normal(size=25))
            direct,adjoint=op.direct(x),op.adjoint(y)
            assert_allclose(direct.dot(y),x.dot(adjoint),atol=1e-12)
            out_y=op.range_geometry().allocate(dtype=np.float64)
            out_x=ig.allocate(dtype=np.float64)
            self.assertIs(op.direct(x,out=out_y),out_y)
            self.assertIs(op.adjoint(y,out=out_x),out_x)
            assert_allclose(out_y.as_array(),direct.as_array())
            assert_allclose(out_x.as_array(),adjoint.as_array())
        ig=ImageGeometry(voxel_num_x=4,voxel_num_y=3)
        ig.set_labels(['horizontal_x','horizontal_y'])
        with self.assertRaises(ValueError): JacobProjector(ig,[[0,0]],[[1,1]])

    def test_tensor_blocks_and_cgls(self):
        from cil.framework import ImageGeometry
        from cil.optimisation.algorithms import CGLS
        from jacobsrays.cil import TensorJacobProjector
        rng=np.random.default_rng(54)
        for dim in [2,3]:
            ig=ImageGeometry(voxel_num_x=4,voxel_num_y=3,voxel_num_z=2 if dim==3 else 0,dtype=np.float64)
            s,e=rng.uniform(-4,4,(2,100,dim))
            op=TensorJacobProjector(ig,s,e,rank=2)
            x=op.domain_geometry().allocate()
            for j in range(op.ncomponents): x.get_item(j).fill(rng.normal(size=ig.shape))
            y=op.range_geometry().allocate(); y.fill(rng.normal(size=100))
            assert_allclose(op.direct(x).dot(y),x.dot(op.adjoint(y)),atol=1e-12)
            out_x=op.domain_geometry().allocate(); out_y=op.range_geometry().allocate()
            self.assertIs(op.direct(x,out=out_y),out_y)
            self.assertIs(op.adjoint(y,out=out_x),out_x)
            assert_allclose(out_y.as_array(),op.direct(x).as_array())
            expected=op.adjoint(y)
            for j in range(op.ncomponents):
                assert_allclose(out_x.get_item(j).as_array(),expected.get_item(j).as_array())
            data=op.direct(x)
            solver=CGLS(initial=op.domain_geometry().allocate(0),operator=op,data=data)
            solver.run(15,verbose=0)
            residual=(op.direct(solver.solution)-data).norm()
            self.assertLess(residual,.1*data.norm())
