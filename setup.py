from setuptools import setup
import re
import os
import subprocess

try:
    long_description = subprocess.check_output(
        'pandoc --to rst README.md', shell=True)
except(IOError, ImportError, subprocess.CalledProcessError):
    long_description = open('README.md').read()

with open('zounds/__init__.py', 'r') as fd:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        fd.read(),
        re.MULTILINE).group(1)

download_url = 'https://github.com/jvinyard/zounds/tarball/{version}' \
    .format(**locals())

on_rtd = os.environ.get('READTHEDOCS') == 'True'

extension_modules = []

if not on_rtd:
    try:
        import numpy as np
        from distutils.extension import Extension

        countbits = Extension(
            name='countbits',
            sources=['zounds/nputil/countbits.pyx'],
            include_dirs=[np.get_include()],
            extra_compile_args=[
                '-c',
                '-mpopcnt',
                '-shared',
                '-pthread',
                '-fPIC',
                '-fwrapv',
                '-O2',
                '-Wall',
                '-fno-strict-aliasing'
            ])
        extension_modules = [countbits]
    except ImportError:
        extension_modules = []

setup(
    name='zounds',
    version=version,
    url='https://github.com/JohnVinyard/zounds',
    author='John Vinyard',
    author_email='john.vinyard@gmail.com',
    long_description=long_description,
    download_url=download_url,
    ext_modules=extension_modules,
    packages=[
        'zounds',
        'zounds.basic',
        'zounds.datasets',
        'zounds.index',
        'zounds.learn',
        'zounds.nputil',
        'zounds.segment',
        'zounds.soundfile',
        'zounds.spectral',
        'zounds.loudness',
        'zounds.synthesize',
        'zounds.timeseries',
        'zounds.ui',
        'zounds.persistence',
        'zounds.core',
        'zounds.util'
    ],
    install_requires=[
        'featureflow',
        'nose',
        'unittest2',
        'certifi==2017.7.27.1',
        'requests',
        'tornado==4.5.3',
        'pysoundfile',
        'matplotlib==1.5.0',
        'argparse',
        'ujson',
        'numpy==1.15.3',
        'scipy==1.1.0',
        'torch==0.4.0'
    ],
    package_data={
        'nputil': ['*.pyx', '*.pyxbld'],
        'ui': ['*.html', '*.js']
    },
    scripts=['bin/zounds-quickstart'],
    include_package_data=True
)
