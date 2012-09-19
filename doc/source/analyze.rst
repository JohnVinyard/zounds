Analyze
======================================
.. currentmodule:: zounds.analyze.extractor

--------------------------------------
The Extractor Abstract Base Class
--------------------------------------
.. autoclass:: Extractor
	:members: __init__,dim,dtype,_process

--------------------------------------
The SingleInput Class
--------------------------------------
.. autoclass:: SingleInput
	:members:

--------------------------------------
Spectral Features
--------------------------------------
.. currentmodule:: zounds.analyze.feature.spectral

.. autoclass:: FFT
	:members: __init__

.. autoclass:: BarkBands
	:members: __init__

.. autoclass:: Loudness
	:members: __init__
	
.. autoclass:: SpectralCentroid
	:members: __init__

.. autoclass:: SpectralFlatness
	:members: __init__

.. autoclass:: BFCC
	:members: __init__

.. autoclass:: Difference
	:members: __init__

.. autoclass:: Flux
	:members: __init__

--------------------------------------
The Composite Extractor
--------------------------------------
.. currentmodule:: zounds.analyze.feature.composite

.. autoclass:: Composite
	:members: __init__