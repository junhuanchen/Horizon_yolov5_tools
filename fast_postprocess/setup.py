# sudo python3 setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# 为编译器获取numpy路径
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# 源文件路径
source_files = ["__RDKyolov5postprocess__.pyx", "RDKyolov5postprocess.cpp"]

# 头文件路径
include_dirs = [numpy_include]

ext_modules = [Extension(
    "RDKyolov5postprocess", 
    source_files, 
    language="c++",
    include_dirs = include_dirs,
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])]#,
              #extra_compile_args=["-O0"],  # 关闭优化
              #extra_link_args=["-O0"]    # 关闭链接时的优化
    #)]

# import name: RDK_yolov5_postprocess
setup(name="RDKyolov5postprocess", ext_modules=cythonize(ext_modules))

print("\033[32mSetup has been completed!\033[0m")