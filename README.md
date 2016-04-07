[![Build Status](https://travis-ci.org/JohnVinyard/zounds.svg?branch=master)](https://travis-ci.org/JohnVinyard/zounds)
[![Coverage Status](https://coveralls.io/repos/github/JohnVinyard/zounds/badge.svg?branch=master)](https://coveralls.io/github/JohnVinyard/zounds?branch=master)
[![PyPI](https://img.shields.io/pypi/v/zounds.svg)](https://pypi.python.org/pypi/zounds)

# Usage
Zounds is a dataflow library for building directed acyclic graphs that transform audio. It uses the 
[featureflow](https://github.com/JohnVinyard/featureflow) library to define the processing pipelines.
  

For example, here's the definition of a pipeline that computes a sliding short-time fourier transform of some audio, 
and then computes spectrograms on the bark and chroma scales.

```python
import featureflow as ff
import zounds

windowing = zounds.HalfLapped()
samplerate = zounds.SR44100()


class Settings(ff.PersistenceSettings):
    id_provider = ff.UuidProvider()
    key_builder = ff.StringDelimitedKeyBuilder()
    database = ff.FileSystemDatabase(path='data', key_builder=key_builder)


class AudioGraph(ff.BaseModel):

    meta = ff.JSONFeature(
        zounds.MetaData,
        encoder=zounds.AudioMetaDataEncoder,
        store=True)

    raw = ff.ByteStreamFeature(
        ff.ByteStream,
        chunksize=2 * 44100 * 30 * 2,
        needs=meta,
        store=False)

    ogg = zounds.OggVorbisFeature(
        zounds.OggVorbis,
        needs=raw,
        store=True)

    pcm = zounds.ConstantRateTimeSeriesFeature(
        zounds.AudioStream,
        needs=raw,
        store=False)

    resampled = zounds.ConstantRateTimeSeriesFeature(
        zounds.Resampler,
        needs=pcm,
        samplerate=samplerate,
        store=False)

    windowed = zounds.ConstantRateTimeSeriesFeature(
        zounds.SlidingWindow,
        needs=resampled,
        wscheme=zounds.HalfLapped(),
        wfunc=zounds.OggVorbisWindowingFunc(),
        store=False)

    fft = zounds.ConstantRateTimeSeriesFeature(
        zounds.FFT,
        needs=windowed,
        store=False)

    bark = zounds.ConstantRateTimeSeriesFeature(
        zounds.BarkBands,
        needs=fft,
        store=True)

    chroma = zounds.ConstantRateTimeSeriesFeature(
        zounds.Chroma,
        needs=fft,
        store=True)

    bfcc = zounds.ConstantRateTimeSeriesFeature(
        zounds.BFCC,
        needs=fft,
        store=True)


class Document(AudioGraph, Settings):
    pass
```

Data can be processed, and later retrieved as follows:

```python
>>> _id = doc = Document.process(meta='https://example.com/audio.wav')
>>> doc = Document(_id)
>>> doc.chroma.shape
(321, 12)
```

# Installation
 
## Libsndfile Issues
Installation currently requires you to build lbiflac and libsndfile from source, because of 
[an outstanding issue](https://github.com/bastibe/PySoundFile/issues/130) that will be corrected when the apt package 
is updated to `libsndfile 1.0.26`.  Download and run 
[this script](https://raw.githubusercontent.com/JohnVinyard/zounds/master/setup.sh) to handle this step.

## Numpy and Scipy
The [Anaconda](https://www.continuum.io/downloads) python distribution is highly recommended.

## Zounds
Finall, just:

```bash
pip install zounds
```