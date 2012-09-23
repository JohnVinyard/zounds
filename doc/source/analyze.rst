Feature Extractors
======================================
.. currentmodule:: zounds.analyze.extractor

--------------------------------------
The Extractor Abstract Base Class
--------------------------------------
.. autoclass:: Extractor
	:members: __init__,dim,dtype,_process
	

--------------------------------------
The ExtractorChain Class
--------------------------------------
.. autoclass:: ExtractorChain
	:members: __init__

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

--------------------------------------
Extracting Learned Features
--------------------------------------
.. currentmodule:: zounds.analyze.feature.learned

.. autoclass:: Learned
	:members: __init__
	
--------------------------------------
Some Basic Operations
--------------------------------------
.. currentmodule:: zounds.analyze.feature.basic

.. autoclass:: Abs
	:members: __init__
	
.. autoclass:: UnitNorm
	:members: __init__

.. autoclass:: Log
	:members: __init__

.. autoclass:: SliceX
	:members: __init__
