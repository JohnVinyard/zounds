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

setup(
        name='zounds',
        version=version,
        url='http://www.johnvinyard.com',
        author='John Vinyard',
        author_email='john.vinyard@gmail.com',
        long_description=long_description,
        download_url=download_url,
        packages=[
            'zounds',
            'zounds.basic',
            'zounds.index',
            'zounds.learn',
            'zounds.nputil',
            'zounds.segment',
            'zounds.soundfile',
            'zounds.spectral',
            'zounds.synthesize',
            'zounds.timeseries',
            'zounds.ui'
        ],
        install_requires=[
            'featureflow',
            'nose',
            'unittest2',
            'requests',
            'tornado',
            'pysoundfile',
            'cython',
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
