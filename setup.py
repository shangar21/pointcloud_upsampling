from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "bolt.bilateral_smooth_src", 
        ["bolt/bilateral_smooth_src/bilateral_smooth.cpp"]
    ),
]

setup(
        name="bolt",
        version="0.1",
        packages=find_packages(include=['bolt', 'bolt.*']),
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        install_requires=[
            'open3d',
            'numpy',
            'tqdm',
            'matplotlib'
        ]
)
