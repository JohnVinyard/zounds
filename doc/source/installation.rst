Installation
=================================

=================================
Caveats
=================================
This is a very early release of Zounds. So far, it has only been tested on Ubuntu 10.10.

=================================
Download
=================================
You can download the latest source distribution here: `zounds-0.02.tar.gz <https://bitbucket.org/jvinyard/zounds2/downloads/zounds-0.02.tar.gz>`_.
For the quickstart tutorial, it's highly recommended that you download the source distribution, and *not* clone the source repository.

.. WARNING::
	Please don't run setup.py right away!  Keep reading...

=================================
Dependencies
=================================
Zounds has a pretty hefty list of dependencies at the moment, but there are resources
included in the source distribution to make installing them as painless as possible.

--------------------------------
Libraries
--------------------------------
There are quite a few libraries you'll need to install, upon which Zounds' python
dependencies rely.  If you're interested in the details, take a look at dependencies.sh
in the source distribution folder for a list of packages, and justifications for each.
Otherwise, run::
	chmod a+x dependencies.sh

so you can execute the script, and then::

	.\dependencies.sh

This will install any libraries you need that you don't already have.

--------------------------------
Numpy and Scipy
--------------------------------
Unfortunately, the setup script doesn't build the numpy and scipy libraries correctly,
so you'll have to do this by hand.  If you attempt to run setup.py before completing
this step, it'll complain. To make it shut up, do::
	sudo pip install numpy
	sudo pip install scipy

in that order.

=================================
Setup.py
=================================
Run::

	sudo python setup.py install 

.. WARNING::
	This step has some problems too. For some reason, both scikits.audiolab and tables 
	(the PyTables package) cause errors that halt the setup script. In both cases, 
	you can simply re-issue the command above, and things will continue along just fine.
	Klunky, but it gets the job done.

While you're still in the source distribution directory, run::

	nosetests

to make sure everything is working.

=================================
Bravo!
=================================
You've succesfully installed zounds! Now on to the :doc:`Quick Start Tutorial </quick-start>`


	


