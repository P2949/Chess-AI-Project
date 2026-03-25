# setup_cython.py — Build the Cython chess acceleration module
#
# Usage:
#     python setup_cython.py build_ext --inplace
#
# Or if you have cythonize:
#     cythonize -i fast_chess.pyx

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "fast_chess",
        sources=["fast_chess.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    )
]

setup(
    name="fast_chess",
    ext_modules=cythonize(extensions, compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "nonecheck": False,
    }),
)
