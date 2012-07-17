from setuptools import setup

npmsg = 'You must have numpy >= 1.6 installed.%s Type "sudo pip install numpy" to install.'
try:
    import numpy
    vparts = (int(s) for s in numpy.__version__.split('.'))
    if vparts[0] < 1 or vparts[1] < 6:
        print npmsg % ('  Yours is version %s' % numpy.__version__)
except ImportError:
    print npmsg % ''
    exit()
    
spmsg = 'You must have scipy installed. Type "sudo pip install scipy" to install.'
try:
    import scipy
except ImportError:
    print spmsg
    exit()

setup(
      name = 'Zounds',
      version = '0.01',
      url = 'http://www.johnvinyard.com',
      author = 'John Vinyard',
      author_email = 'john.vinyard@gmail.com',
      packages = ['','acquire','analyze','data','learn',
                  'model','nputil','queue','visualize'],
      install_requires = ['tables','cython','numexpr',
                          'nose','scikits.audiolab',
                          'matplotlib','scipy','numpy']
)