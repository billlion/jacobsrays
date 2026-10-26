# SPDX-License-Identifier: Apache-2.0
import os
import sys
import numpy
from setuptools import Extension, setup

flags = ["/O2", "/std:c++14"] if sys.platform == "win32" else ["-O3", "-std=c++11"]
link = []
if os.environ.get("JACOBSRAYS_OPENMP") == "1":
    flags += ["/openmp"] if sys.platform == "win32" else ["-fopenmp"]
    if sys.platform != "win32":
        link += ["-fopenmp"]

setup(ext_modules=[Extension("jacobsrays._native", ["src/python.cpp"],
    include_dirs=[numpy.get_include(), "include"], language="c++",
    extra_compile_args=flags, extra_link_args=link)])
