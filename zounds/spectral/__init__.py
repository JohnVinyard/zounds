"""
The spectral module contains classes that aid in dealing with frequency-domain
representations of sound
"""

from sliding_window import \
    SlidingWindow, OggVorbisWindowingFunc, NDSlidingWindow, WindowingFunc, \
    HanningWindowingFunc

from spectral import \
    FFT, DCT, DCTIV, MDCT, BarkBands, Chroma, BFCC, SpectralCentroid, \
    SpectralFlatness, FrequencyAdaptiveTransform, FrequencyWeighting

from tfrepresentation import FrequencyDimension, ExplicitFrequencyDimension

from weighting import AWeighting

from frequencyscale import \
    LinearScale, LogScale, FrequencyBand, FrequencyScale, GeometricScale, \
    ExplicitScale

from frequencyadaptive import FrequencyAdaptive