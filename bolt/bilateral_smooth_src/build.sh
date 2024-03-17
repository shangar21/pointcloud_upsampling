#!/bin/bash
# This script is used to build the project

g++ -O3 -Wall -shared -std=c++20 -fPIC $(python -m pybind11 --includes) bilateral_smooth.cpp -o bilateral_smooth$(python3-config --extension-suffix)
