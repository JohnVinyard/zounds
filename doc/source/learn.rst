The Learn Module
====================================
.. currentmodule:: zounds.learn

.. automodule:: zounds.learn

The Pipeline Class
-----------------------------------
.. currentmodule:: zounds.model.pipeline

.. autoclass:: Pipeline
	:members: __init__,train,__call__

Fetch and Derived Classes
-----------------------------------
.. currentmodule:: zounds.learn.fetch

.. autoclass:: Fetch
	:members:  __call__

.. autoclass:: NoOp

Fetching Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: PrecomputedFeature
	:members: __init__

Fetching Patches of Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sometimes, it will be desirable to sample randomly not only in time, but in 
feature-space as well.  Concretely, consider a case where we'd like to train
a learning algorithm on constant-sized patches that are drawn randomly from 
spectograms, both in time and frequency.

This section will introduce a couple helper classes that build towards that goal,
and finally, the :py:class:`PrecomputedPatch` class, which makes randomly sampling
from both time and feature-space possible.

.. autoclass:: Patch
	:members: __init__

.. autoclass:: NDPatch
	:members: __init__

.. autoclass:: PrecomputedPatch
	:members: __init__