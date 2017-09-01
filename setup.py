from setuptools import setup
import re



try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

with open('zounds/__init__.py', 'r') as fd:
    version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            fd.read(),
            re.MULTILINE).group(1)

download_url = 'https://github.com/jvinyard/zounds/tarball/{version}'\
    .format(**locals())

try:
    import numpy as np
    from distutils.extension import Extension

    countbits = Extension(
        name='countbits',
        sources=['zounds/nputil/countbits.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=[
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
            'requests',
            'tornado',
            'pysoundfile',
            'matplotlib',
            'argparse'
        ],
        package_data={
            'nputil': ['*.pyx', '*.pyxbld'],
            'ui': ['*.html', '*.js']
        },
        scripts=['bin/zounds-quickstart'],
        include_package_data=True
)
