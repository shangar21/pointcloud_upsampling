from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess
import os

def build_mls():
    os.chdir('pcl_src')
    os.makedirs('build', exist_ok=True)
    os.chdir('build')
    subprocess.run(['cmake', '..'])
    subprocess.run(['cmake', '--build', '.'])

install_requires = open('requirements.txt').read().strip().split('\n')

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
        install_requires=install_requires
)

build_mls()
