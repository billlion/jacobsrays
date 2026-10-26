import importlib.util
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm
from jacobsrays import Grid, trace

spec = importlib.util.spec_from_file_location('transport_example', Path(__file__).parents[1]/'examples/nonabelian_transport.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


class TransportTests(unittest.TestCase):
    def test_noncommuting_order_and_connection_reversal(self):
        grid=Grid((2,1))
        A=np.array([[0,1j],[1j,0]])
        B=np.array([[0,1],[-1,0]])
        values=np.array([[A,B]])
        p=trace(grid,[-1,.5],[3,.5])
        U=transport.ordered_transport(p,values)
        assert_allclose(U,expm(B)@expm(A),atol=1e-14)
        self.assertGreater(np.linalg.norm(U-expm(A+B)),.1)
        reverse=trace(grid,[3,.5],[-1,.5])
        V=transport.ordered_transport(reverse,-values)
        assert_allclose(V@U,np.eye(2),atol=1e-14)
        assert_allclose(U.conj().T@U,np.eye(2),atol=1e-14)
        miss=trace(grid,[-1,3],[3,3])
        assert_allclose(transport.ordered_transport(miss,values),np.eye(2))
