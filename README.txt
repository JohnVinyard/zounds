TODO: Description of Zounds

Installation
===============================================================================
1) Run the dependencies.sh script. This will install packages needed to build
and run Zounds' python dependencies.  Take a look at dependencies.sh to see the
full list of packages, as well as justifications for each.

2) Unfortunately, setup.py has trouble installing numpy and scipy, so these should
be installed manually prior to running setup.py. Type "sudo pip install numpy"
and "sudo pip install scipy", in that order, to install them.

3) Run "sudo python setup.py install". This step has some problems too. For some
reason, scikits.audiolab and table (the PyTables package), both cause errors
that halt the setup script. In both cases, you can simply re-issue the
"sudo python setup.py install" command, and things will continue along just fine.
Your guess is as good as mine.

4) Finally, while you're still in Zounds source distribution directory, run 
"nosetests" to ensure everything is working properly.