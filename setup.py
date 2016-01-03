from setuptools import setup
import os


def read(fname):
    """
    This is yanked from the setuptools documentation at
    http://packages.python.org/an_example_pypi_project/setuptools.html. It is
    used to read the text from the README file.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='zounds',
    version='0.02',
    url='http://www.johnvinyard.com',
    author='John Vinyard',
    author_email='john.vinyard@gmail.com',
    long_description=read('README.md'),
    packages=['zounds', 'zounds.node', 'zounds.learn', 'zounds.learn.nnet',
              'zounds.nputil', 'zounds.analyze', 'zounds.visualize'],
    install_requires=['nose', 'unittest2', 'requests', 'tornado'],
    package_data={'nputil': ['*.pyx', '*.pyxbld']},
    include_package_data=True
)


# import sys
# import os
# import string
# import subprocess

# KLUDGE: Is there a better way to get setuptools commands?
# install = 'install' in sys.argv[1:]


# def run_command(cmd):
#     p = subprocess.Popen(cmd, shell=True)
#     rc = p.wait()
#     if rc:
#         # KLUDGE: What should I do here?
#         raise RuntimeError()


# if install:
#     # make dependencies.sh executable
#     run_command('chmod a+x dependencies.sh')
#     # install non-python dependencies
#     run_command('./dependencies.sh')

# At this point, setuptools should be available
# from setuptools import setup
#
# if install:
#     # install numpy
#     run_command('pip install numpy')
#     # install scipy
#     run_command('pip install scipy')


# def setup_jack_audio():
#     # KLUDGE: This is a hack. Right?  I'd like to add the currently logged-in user
#     # to the "audio" group, since JACK's installation has already setup realtime
#     # audio permissions for users in that group, but the user is impersonating
#     # root, so I'm using the "logname" command to guess what the user's name
#     # *probably* is.
#
#     fail_msg = 'There was a problem adding you to the audio user group : %s'
#
#     p = subprocess.Popen( \
#             'logname', shell=True, stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE)
#     rc = p.wait()
#     if rc:
#         print fail_msg % p.stderr.read()
#         return
#
#     # the output of logname ends with a newline
#     username = p.stdout.read()[:-1]
#     print 'adding %s to the audio user group' % username
#     # Add the current user to the audio group
#     p = subprocess.Popen('usermod -a -G audio %s' % username, shell=True)
#     rc = p.wait()
#     if rc:
#         print fail_msg % p.stderr.read()


# build up the list of packages
# packages = []
# root_dir = os.path.dirname(__file__)
# if root_dir != '':
#     os.chdir(root_dir)
# zounds_dir = 'zounds'
#
# for dirpath, dirnames, filenames in os.walk(zounds_dir):
#     if '__init__.py' in filenames:
#         pathparts = dirpath.split(os.path.sep)
#         index = pathparts.index(zounds_dir)
#         packages.append(string.join(pathparts[index:], '.'))
#
#

#
#
# python_packages = ['bitarray', 'tables', 'nose', 'scikits.audiolab',
#                    'matplotlib', 'web.py', 'scipy', 'numpy', 'unittest2',
#                    'numexpr', 'cython', 'pysoundfile','requests', 'flow']
#
# # argparse was introduced into the standard library in python 2.7. Instead of
# # checking the python version, just try to import it. If the import fails, add
# # argparse to the list of dependencies
# try:
#     import argparse
# except ImportError:
#     python_packages.append('argparse')
#
# c_ext = ['*.c', '*.h']
# pyx_ext = ['*.pyx', '*.pyxbld']
#
# setup(
#         name='zounds',
#         version='0.03',
#         url='http://www.johnvinyard.com',
#         author='John Vinyard',
#         author_email='john.vinyard@gmail.com',
#         long_description=read('README.txt'),
#         scripts=['zounds/quickstart/zounds-quickstart.py',
#                  'zounds/zounds-audio-test.py'],
#         package_data={'quickstart': ['*.py'],
#                       'quickstart/websearch': ['*.py'],
#                       'quickstart/websearch/css': ['*.css'],
#                       'quickstart/websearch/js': ['*.js'],
#                       'quickstart/websearch/templates': ['*.html'],
#                       'pattern': c_ext + pyx_ext,
#                       'nputil': pyx_ext},
#         include_package_data=True,
#         packages=packages,
#         install_requires=python_packages
# )
#
# if install:
#     # add the current user to the audio user group
#     setup_jack_audio()
#     # run the unit tests
#     run_command('nosetests')
#     print 'Done! Enjoy Zounds.'
#
# import os
# from setuptools import setup
#
#
# def read(fname):
#     """
#     This is yanked from the setuptools documentation at
#     http://packages.python.org/an_example_pypi_project/setuptools.html. It is
#     used to read the text from the README file.
#     """
#     return open(os.path.join(os.path.dirname(__file__), fname)).read()
