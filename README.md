[![Build Status](https://travis-ci.org/JohnVinyard/zounds.svg?branch=master)](https://travis-ci.org/JohnVinyard/zounds)
[![Coverage Status](https://coveralls.io/repos/github/JohnVinyard/zounds/badge.svg?branch=master)](https://coveralls.io/github/JohnVinyard/zounds?branch=master)
[![PyPI](https://img.shields.io/pypi/v/zounds.svg)](https://pypi.python.org/pypi/zounds)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat&maxAge=86400)](http://zounds.readthedocs.io/en/latest/?badge=latest)

# Motivation

Zounds is a python library for working with sound.  Its primary goals are to:

- layer semantically meaningful audio manipulations on top of numpy arrays
- [help to organize the definition and persistence of audio processing
  pipelines and machine learning experiments with sound](https://github.com/JohnVinyard/zounds/tree/master/zounds/learn)

Audio processing graphs and machine learning pipelines are defined using
[featureflow](https://github.com/JohnVinyard/featureflow).

# A Quick Example

```python
import zounds

Resampled = zounds.resampled(resample_to=zounds.SR11025())


@zounds.simple_in_memory_settings
class Sound(Resampled):
    """
    A simple pipeline that computes a perceptually weighted modified discrete
    cosine transform, and "persists" feature data in an in-memory store.
    """

    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        needs=Resampled.resampled,
        wscheme=zounds.HalfLapped(),
        wfunc=zounds.OggVorbisWindowingFunc(),
        store=True)

    mdct = zounds.ArrayWithUnitsFeature(
        zounds.MDCT,
        needs=windowed)

    weighted = zounds.ArrayWithUnitsFeature(
        lambda x: x * zounds.AWeighting(),
        needs=mdct)

if __name__ == '__main__':

    # produce some audio to test our pipeline, and encode it as FLAC
    synth = zounds.SineSynthesizer(zounds.SR44100())
    samples = synth.synthesize(zounds.Seconds(5), [220., 440., 880.])
    encoded = samples.encode(fmt='FLAC')

    # process the audio, and fetch features from our in-memory store
    _id = Sound.process(meta=encoded)
    sound = Sound(_id)

    # grab all the frequency information, for a subset of the duration
    start = zounds.Milliseconds(500)
    end = start + zounds.Seconds(2)
    snippet = sound.weighted[start: end, :]

    # grab a subset of frequency information for the duration of the sound
    freq_band = slice(zounds.Hertz(400), zounds.Hertz(500))
    a440 = sound.mdct[:, freq_band]

    # produce a new set of coefficients where only the 440hz sine wave is
    # present
    filtered = sound.mdct.zeros_like()
    filtered[:, freq_band] = a440

    # apply a geometric scale, which more closely matches human pitch
    # perception, and apply it to the linear frequency axis
    scale = zounds.GeometricScale(50, 4000, 0.05, 100)
    log_coeffs = scale.apply(sound.mdct, zounds.HanningWindowingFunc())

    # reconstruct audio from the MDCT coefficients
    mdct_synth = zounds.MDCTSynthesizer()
    reconstructed = mdct_synth.synthesize(sound.mdct)
    filtered_reconstruction = mdct_synth.synthesize(filtered)

    # start an in-browser REPL that will allow you to listen to and visualize
    # the variables defined above (and any new ones you create in the session)
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.weighted,
        globals=globals(),
        locals=locals())
    app.start(9999)
```

Find more inspiration in the [examples folder](https://github.com/JohnVinyard/zounds/tree/master/examples),
or on the [blog](http://johnvinyard.github.io/).

# Installation
 
## Libsndfile Issues
Installation currently requires you to build lbiflac and libsndfile from source, because of 
[an outstanding issue](https://github.com/bastibe/PySoundFile/issues/130) that will be corrected when the apt package 
is updated to `libsndfile 1.0.26`.  Download and run 
[this script](https://raw.githubusercontent.com/JohnVinyard/zounds/master/setup.sh) to handle this step.

## Numpy and Scipy
The [Anaconda](https://www.continuum.io/downloads) python distribution is highly recommended.

## Zounds
Finally, just:

```bash
pip install zounds
```