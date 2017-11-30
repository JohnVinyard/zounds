"""
The `learn` module includes classes that make it possible to define processing
graphs whose leaves are trained machine learning models.

While much of :module:`zounds.soundfile`, :module:`zounds.spectral`, and
:module:`zounds.timeseries` focus on processing nodes that can be composed into
a processing graph to extract features from a single piece of audio, the `learn`
module focuses on defining graphs that extract features or trained models from
an entire corpus of audio.
"""

from learn import KMeans, Learned

from preprocess import \
    MeanStdNormalization, UnitNorm, Log, PreprocessingPipeline, Multiply, \
    Slicer, InstanceScaling, Reshape, Weighted, MuLawCompressed, SimHash, \
    AbsoluteValue, Binarize, Sharpen

from sklearn_preprocessor import SklearnModel, WithComponents

from pytorch_model import \
    PyTorchAutoEncoder, PyTorchGan, PyTorchNetwork, GanTrainer, \
    WassersteinGanTrainer, SupervisedTrainer

from random_samples import ReservoirSampler, ShuffledSamples

from template_match import TemplateMatch

from util import simple_settings

try:
    from util import Conv1d, ConvTranspose1d, Conv2d, ConvTranspose2d
except ImportError:
    pass


