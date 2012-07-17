from setuptools import setup

# TODO: This code should only run when the 'install' command is used.
npmsg = 'You must have numpy >= 1.6 installed.%s Type "sudo pip install numpy" to install.'
try:
    import numpy
    vparts = [int(s) for s in numpy.__version__.split('.')]
    if vparts[0] < 1 or vparts[1] < 6:
        print npmsg % ('  Yours is version %s' % numpy.__version__)
        exit()
    print 'You\'ve got numpy version %s. Great!' % numpy.__version__
except ImportError:
    print npmsg % ''
    exit()
    
spmsg = 'You must have scipy installed. Type "sudo pip install scipy" to install.'
try:
    import scipy
    print 'You\'ve got scipy version %s. Great!' % scipy.__version__
except ImportError:
    print spmsg
    exit()

import os
import string
# build up the list of packages
packages = []
root_dir = os.path.dirname(__file__)
if root_dir != '':
    os.chdir(root_dir)
zounds_dir = 'zounds'

for dirpath, dirnames, filenames in os.walk(zounds_dir):
    if '__init__.py' in filenames:
        pathparts = dirpath.split(os.path.sep)
        index = pathparts.index(zounds_dir)
        packages.append(string.join(pathparts[index:],'.'))


def read(fname):
    '''
    This is yanked from the setuptools documentation at 
    http://packages.python.org/an_example_pypi_project/setuptools.html. It is
    used to read the text from the README file.
    '''
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
      name = 'zounds',
      version = '0.01',
      url = 'http://www.johnvinyard.com',
      author = 'John Vinyard',
      author_email = 'john.vinyard@gmail.com',
      long_description = read('README.txt'),
      scripts = ['zounds/quickstart/quickstart.py'],
      packages = packages,
      install_requires = ['tables','cython','numexpr',
                          'nose','scikits.audiolab',
                          'matplotlib','scipy','numpy']
)