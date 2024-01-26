# setup文件
# Horizon_yolov5tools/目录下命令行输入 sudo python3 setup.py build_ext --inplace 以编译为库文件
# numpy API提示的waring可忽略
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# 为编译器获取numpy路径
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

source_files = ["__postprocess__.pyx"]
ext_modules = [Extension("postprocess", source_files, language="c++",include_dirs = [numpy_include])]

# import name: fast_postprocess
setup(name="postprocess", ext_modules=cythonize(ext_modules))

print("\033[32mSetup has been completed!\033[0m")