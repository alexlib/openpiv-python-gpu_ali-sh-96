# OpenPIV-Python-GPU
Algorithms for PIV image processing with a GPU.

[![DOI](https://zenodo.org/badge/670884759.svg)](https://zenodo.org/badge/latestdoi/670884759)

[openpiv-python](https://github.com/OpenPIV/openpiv-python) consists of Python modules for performing particle image velocimetry (PIV) analysis on a set of image pairs. [openpiv-python-gpu](https://github.com/ali-sh-96/openpiv-python-gpu) is a GPU implementation of the same algorithms as [OpenPIV-Python-CPU](https://github.com/ali-sh-96/openpiv-python-cpu), depending only on CuPy for GPU acceleration. The objective of this project was to reduce the PIV computation time and maintain compatibility with the CPU-based version.

## Warning
OpenPIV-Python is currently under active development, which means it might contain some bugs, and its API is subject to change. The algorithms have been tested only on Windows (work station and laptops) at this time.

## Installation
Use the following command to install from GitHub:

    pip install git+https://github.com/ali-sh-96/openpiv-python-gpu

## Documentation
The OpenPIV documentation is readily accessible on the project's webpage at https://openpiv.readthedocs.org. For information on how to use the modules, see the tutorial notebooks below.

## Tutorials
- [Basic tutorial](https://colab.research.google.com/github/ali-sh-96/openpiv-python-gpu/blob/main/openpiv_gpu/tutorials/openpiv_python_gpu_tutorial.ipynb)
- [Advanced tutorial](https://colab.research.google.com/github/ali-sh-96/openpiv-python-gpu/blob/main/openpiv_gpu/tutorials/openpiv_python_gpu_advanced_tutorial.ipynb)

## Contributors
1. [OpenPIV team](https://groups.google.com/forum/#!forum/openpiv-users)
2. [Alex Liberzon](https://github.com/alexlib)
3. [Ali Shirinzad](https://github.com/ali-sh-96)
4. Pierre E. Sullivan

Copyright statement: `gpu_smoothn.py` is the GPU accelerated version of `cpu_smoothn.py`, which it self is a Python version of `smoothn.m` originally created by
[D. Garcia](https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn), written by Prof. Lewis, and available on
[GitHub](https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py). We are thankful to the original authors for
releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the
authors regarding their license.

## How to cite this work
Shirinzad, A., Jaber, K., Xu, K., & Sullivan, P. E. (2023). An Enhanced Python-Based Open-Source Particle Image Velocimetry Software for Use with Central Processing Units. Fluids, 8(11), 285. https://doi.org/10.3390/fluids8110285