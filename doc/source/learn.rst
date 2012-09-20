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

Preprocess
------------------------------------------
.. currentmodule:: zounds.learn.preprocess

.. autoclass:: Preprocess
	:members: _preprocess

.. autoclass:: NoOp

.. autoclass:: SubtractMean
	:members: __init__

.. autoclass:: DivideByStd
	:members: __init__

.. autoclass:: UnitNorm

.. autoclass:: SequentialPreprocessor
	:members: __init__

.. autoclass:: Downsample
	:members: __init__

Learn
-----------------------------------------
.. currentmodule:: zounds.learn.learn

.. autoclass:: Learn
	:members: train, __call__

Clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: zounds.learn.cluster.kmeans

.. autoclass:: KMeans
	:members: __init__,train,__call__

.. autoclass:: SoftKMeans
	:members: __call__

.. currentmodule:: zounds.learn.cluster.som

.. autoclass:: Som
	:members: __init__,train,__call__

Neural Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: zounds.learn.nnet.rbm

.. autoclass:: Rbm
	:members: __init__,indim,hdim,train

.. autoclass:: LinearRbm
	:members: __init__

.. currentmodule:: zounds.learn.nnet.autoencoder

.. autoclass:: Autoencoder
	:members: __init__,train,__call__

Hashing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: zounds.learn.hash.minhash

.. autoclass:: MinHash
	:members: __init__,train,__call__

	