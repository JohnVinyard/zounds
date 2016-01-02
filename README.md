# Usage
Zounds is a dataflow library for building directed acyclic graphs that transform audio.  For example, here's the definition of a pipeline that computes a sliding short-time fourier transform of some audio, and then computes spectrograms on the bark and chroma scales.

```
windowing_scheme = HalfLapped()
samplerate = SR44100()

class Settings(PersistenceSettings):
    id_provider = UuidProvider()
    key_builder = StringDelimitedKeyBuilder()
    database = InMemoryDatabase(key_builder=key_builder)

class Document(BaseModel, Settings):

    meta = JSONFeature(
        MetaData,
        store = True,
        encoder = AudioMetaDataEncoder)

    raw = ByteStreamFeature(
        ByteStream,
        chunksize=2 * 44100 * 30 * 2,
        store=True)

    pcm = ConstantRateTimeSeriesFeature(
        AudioStream,
        needs=raw,
        store=False)

    resampled = ConstantRateTimeSeriesFeature(
        Resampler,
        needs=pcm,
        samplerate=samplerate,
        store=False)

    windowed = ConstantRateTimeSeriesFeature(
        SlidingWindow,
        needs=resampled,
        wscheme=windowing_scheme,
        wfunc=OggVorbisWindowingFunc(),
        store=False)

    fft = ConstantRateTimeSeriesFeature(
        FFT,
        needs=windowed,
        store=True)

    chroma = ConstantRateTimeSeriesFeature(
        Chroma,
        needs=fft,
        samplerate=samplerate,
        store=True)

    bark = ConstantRateTimeSeriesFeature(
        BarkBands,
        needs=fft,
        samplerate=samplerate,
        store=True)
```

Data can be processed, and later retrieved as follows:

```
>>> import requests
>>> req = requests.Request(method = 'GET', url = 'https://example.com/audio.wav')
>>> _id = doc = Document.process(raw=req)
>>> doc = Document(_id)
>>> doc.chroma.shape
(321, 12)
```
# Installation
## Numpy and Scipy
The [Anaconda](https://www.continuum.io/downloads) python distribution is highly recommended
## PySoundFile
`libsndfile 1.0.26` is required.  Ubuntu 14.04 is still on `libsndfile 1.0.25`.  This means that there are some extra steps involved to get PySoundfile working.

- Ensure that you've got all the dependencies here (https://github.com/erikd/libsndfile). Flac must be >= 1.3.1, so you can't use the package manager (yet)
- Build [libsndfile(https://github.com/erikd/libsndfile) from source
- Get the source for [PySoundfile](https://github.com/bastibe/PySoundFile)
- modify the code to load the libsndfile library from `/usr/local/lib/libsndfile.so`
- install PySoundfile (`python setup.py install`)
