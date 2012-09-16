from setuptools import setup,Extension
import os
import string
import subprocess
import getpass
import sys

# KLUDGE: Is there a better way to get setuptools commands?

install = 'install' in sys.argv[1:]

def force_manual_numpy_scipy_install():
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

def setup_jack_audio():
    
    # KLUDGE: This is a hack. Right?  I'd like to add the currently logged-in user
    # to the "audio" group, since JACK's installation has already setup realtime
    # audio permissions for users in that group, but the user is impersonating
    # root, so I'm using the "logname" command to guess what the user's name
    # *probably* is.
    
    fail_msg = 'There was a problem adding you to the audio user group : %s' 
    
    p = subprocess.Popen(\
     'logname',shell = True,stdout = subprocess.PIPE,stderr = subprocess.PIPE)
    rc = p.wait()
    if rc:
        print fail_msg % p.stderr.read()
        return
    
    # the output of logname ends with a newline
    username = p.stdout.read()[:-1]
    print 'adding %s to the audio user group' % username 
    # Add the current user to the audio group
    p = subprocess.Popen('usermod -a -G audio %s' % username,shell = True)
    rc = p.wait()
    if rc:
        print fail_msg % p.stderr.read()
    
    
    

if install:
    force_manual_numpy_scipy_install()

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

python_packages = ['bitarray','tables','cython','numexpr',
                   'nose','scikits.audiolab',
                   'matplotlib','web.py','scipy','numpy']

# argparse was introduced into the standard library in python 2.7. Instead of
# checking the python version, just try to import it. If the import fails, add
# argparse to the list of dependencies
try:
    import argparse
except ImportError:
    python_packages.append('argparse')

c_ext = ['*.c','*.h']
pyx_ext = ['*.pyx','*.pyxbld']
setup(
      name = 'zounds',
      version = '0.02',
      url = 'http://www.johnvinyard.com',
      author = 'John Vinyard',
      author_email = 'john.vinyard@gmail.com',
      long_description = read('README.txt'),
      scripts = ['zounds/quickstart/zounds-quickstart.py'],
      package_data = {'quickstart' : ['*.py'],
                      'quickstart/websearch'     : ['*.py'],
                      'quickstart/websearch/css' : ['*.css'],
                      'quickstart/websearch/js'  : ['*.js'],
                      'quickstart/websearch/templates' : ['*.html'],
                      'pattern' : c_ext + pyx_ext,
                      'nputil' : pyx_ext},
      include_package_data = True,
      packages = packages,
      install_requires = python_packages
)

if install:
    setup_jack_audio()
    print 'Done! Enjoy Zounds.'