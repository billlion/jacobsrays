"""Compare compiled traversal kernels; requires a separate legacy source checkout.

python benchmarks/compare_legacy.py /path/to/LRTinMatlab/jacobsrays

No legacy code is copied into this project. Empty MEX header stubs allow the
legacy traversal (which does not call MEX itself) to compile in a temp dir.
The test uses full exterior-to-exterior rays, avoiding known legacy defects.
It compares one-pass index/length output, NOT Python or MATLAB wrapper speed.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[1]
legacy=Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as folder:
    tmp=Path(folder)
    for name in ['mex.h','matrix.h']: (tmp/name).write_text('')
    subprocess.run([os.getenv('CC','gcc'),'-O3','-I'+str(tmp),'-I'+str(legacy),'-c',str(legacy/'mod_jacobs_rays.c'),'-o',str(tmp/'old.o')],check=True)
    subprocess.run([os.getenv('CXX','g++'),'-O3','-std=c++11','-I'+str(tmp),'-I'+str(legacy),'-I'+str(root/'include'),str(root/'benchmarks/compare_legacy.cpp'),str(tmp/'old.o'),'-o',str(tmp/'compare')],check=True)
    subprocess.run([str(tmp/'compare')],check=True)
