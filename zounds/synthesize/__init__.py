"""
The `synthesize` module includes classes that can produce audio.  Some, like
:class:`SineSynthesize` can produce simple signals from scratch that are often
useful for test-cases, while others are able to invert common frequency-domain
transforms, like the :class:`MDCTSynthesizer`
"""

from synthesize import \
    FFTSynthesizer, DCTSynthesizer, TickSynthesizer, NoiseSynthesizer, \
    SineSynthesizer, DCTIVSynthesizer, MDCTSynthesizer, \
    FrequencyAdaptiveFFTSynthesizer, FrequencyAdaptiveDCTSynthesizer, \
    SilenceSynthesizer, WindowedAudioSynthesizer, \
    FrequencyDecompositionSynthesizer
