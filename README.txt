About
===============================================================================
Zounds is a python library designed to make experimenting with audio feature
extraction easy.  It allows you to define your features-of-interest in an 
inuitive, pythonic way, store them, and search them.

Zounds is a Python library designed to make prototyping machine listening pipelines
easy!  It allows you to define you features-of-interest in an intuitive, pythonic
way, store them, and search them.

    class FrameModel(Frames):
        fft = Feature(FFT, store = False)
        bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)
        loudness = Feature(Loudness, needs = bark)
        centroid = Feature(SpectralCentroid, needs = bark)
        flatness = Feature(SpectralFlatness, needs = bark) 

Installation
===============================================================================

1. Get the [latest stable release](https://bitbucket.org/jvinyard/zounds2/downloads/zounds-0.03.tar.gz)
2. Run `setup.py`. 