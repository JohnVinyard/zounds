# Usage
Zounds is a dataflow library for building directed acyclic graphs that transform audio. It uses the 
[flow](https://bitbucket.org/jvinyard/flow/) library to define the processing pipelines.
  

For example, here's the definition of a pipeline that computes a sliding short-time fourier transform of some audio, 
and then computes spectrograms on the bark and chroma scales.

```
import flow
import zounds

windowing = zounds.HalfLapped()
samplerate = zounds.SR44100()


class Settings(flow.PersistenceSettings):
    id_provider = flow.UuidProvider()
    key_builder = flow.StringDelimitedKeyBuilder()
    database = flow.FileSystemDatabase(path='data', key_builder=key_builder)


class AudioGraph(flow.BaseModel):

    meta = flow.JSONFeature(
        zounds.MetaData,
        encoder=zounds.AudioMetaDataEncoder,
        store=True)

    raw = flow.ByteStreamFeature(
        flow.ByteStream,
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

```
>>> import requests
>>> req = requests.Request(method = 'GET', url = 'https://example.com/audio.wav')
>>> _id = doc = Document.process(meta=req)
>>> doc = Document(_id)
>>> doc.chroma.shape
(321, 12)
```
# Installation
## Flow

## Numpy and Scipy
The [Anaconda](https://www.continuum.io/downloads) python distribution is highly recommended
## PySoundFile
`libsndfile 1.0.26` is required.  Ubuntu 14.04 is still on `libsndfile 1.0.25`.  This means that there are some extra steps involved to get PySoundfile working.

- Ensure that you've got all the dependencies here (https://github.com/erikd/libsndfile). Flac must be >= 1.3.1, so you can't use the package manager (yet)
- Build [libsndfile(https://github.com/erikd/libsndfile) from source
- Get the source for [PySoundfile](https://github.com/bastibe/PySoundFile)
- modify the code to load the libsndfile library from `/usr/local/lib/libsndfile.so`
- install PySoundfile (`python setup.py install`)
