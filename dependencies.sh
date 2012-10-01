#!/bin/bash
# Python headers needed to build Numpy/Scipy
# Numerical libraries needed to build Numpy/Scipy
# Fortran compiler needed to compile some libraries used by Numpy/Scipy
# Library used by scikits.audiolab to read audio files
# Library used by scikits.audiolab to play sounds
# Zounds uses this library to do resampling
# HDF5 libraries needed for PyTables
# LZO Compression library used by PyTables
# Tool to install python libraries from the Python Package Index
# The following two libraries are needed by matplotlib
# g++ is required to compile Scipy
# jackd1 and libjack-dev are required to use/build against the jack library

sudo apt-get install \
python-dev \
libblas-dev \
liblapack-dev \
gfortran \
libsndfile1-dev \
libasound2-dev \
libsamplerate-dev \
libhdf5-serial-1.8.4 \
libhdf5-serial-dev \
liblzo2-dev \
python-pip \
libfreetype6-dev \
libpng-dev \
g++ \
jackd \
libjack-dev \