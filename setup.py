import distutils.core
import Cython.Build

import Cython.Build
distutils.core.setup(
    ext_modules = Cython.Build.cythonize("test_cython.pyx"))