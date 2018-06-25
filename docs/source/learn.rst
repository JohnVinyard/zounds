Learn
=====
.. automodule:: zounds.learn
.. currentmodule:: zounds.learn

The Basics
----------
.. autoclass:: PreprocessingPipeline
.. autoclass:: Pipeline
.. autoclass:: Preprocessor
    :members:
.. autoclass:: PreprocessResult
.. autoclass:: PipelineResult

Data Preparation
----------------
.. autoclass:: UnitNorm
.. autoclass:: MuLawCompressed
.. autoclass:: MeanStdNormalization
.. autoclass:: InstanceScaling
.. autoclass:: Weighted

Sampling
--------
.. autoclass:: ShuffledSamples

Machine Learning Models
-----------------------
.. autoclass:: KMeans
.. autoclass:: SklearnModel
.. autoclass:: PyTorchNetwork
.. autoclass:: PyTorchGan
.. autoclass:: PyTorchAutoEncoder
.. autoclass:: SupervisedTrainer
.. autoclass:: TripletEmbeddingTrainer
.. autoclass:: WassersteinGanTrainer

Hashing
-------
.. autoclass:: SimHash

Learned Models in Audio Processing Graphs
-----------------------------------------
.. autoclass:: Learned