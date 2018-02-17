"""
The spectral module contains classes that aid in dealing with frequency-domain
representations of sound
"""

from sliding_window import \
    SlidingWindow, OggVorbisWindowingFunc, WindowingFunc, HanningWindowingFunc

from spectral import \
    FFT, DCT, DCTIV, MDCT, BarkBands, Chroma, BFCC, SpectralCentroid, \
    SpectralFlatness, FrequencyAdaptiveTransform, FrequencyWeighting

from tfrepresentation import FrequencyDimension, ExplicitFrequencyDimension

from weighting import AWeighting

from frequencyscale import \
    LinearScale, FrequencyBand, FrequencyScale, GeometricScale, ExplicitScale, \
    Hertz, BarkScale, MelScale, ChromaScale

from frequencyadaptive import FrequencyAdaptive

from functional import \
    fft, stft, apply_scale, frequency_decomposition, phase_shift, rainbowgram, \
    dct_basis
