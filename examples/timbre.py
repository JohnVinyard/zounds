import featureflow as ff
import zounds
import requests
import os
from urlparse import urlparse


class Settings(ff.PersistenceSettings):
    id_provider = ff.UuidProvider()
    key_builder = ff.StringDelimitedKeyBuilder()
    database = ff.LmdbDatabase(path='timbre', key_builder=key_builder)


windowing = zounds.HalfLapped()
STFT = zounds.stft(resample_to=zounds.SR22050(), wscheme=windowing)


class WithTimbre(STFT, Settings):
    bark = zounds.ConstantRateTimeSeriesFeature(
            zounds.BarkBands,
            needs=STFT.fft,
            store=True)

    bfcc = zounds.ConstantRateTimeSeriesFeature(
            zounds.BFCC,
            needs=bark,
            store=True)


@zounds.simple_settings
class BfccKmeans(ff.BaseModel):
    docs = ff.Feature(
            ff.IteratorNode,
            store=False)

    shuffle = ff.NumpyFeature(
            zounds.ReservoirSampler,
            nsamples=1e6,
            needs=docs,
            store=True)

    unitnorm = ff.PickleFeature(
            zounds.UnitNorm,
            needs=shuffle,
            store=False)

    kmeans = ff.PickleFeature(
            zounds.KMeans,
            centroids=128,
            needs=unitnorm,
            store=False)

    pipeline = ff.PickleFeature(
            zounds.PreprocessingPipeline,
            needs=(unitnorm, kmeans),
            store=True)


class WithCodes(WithTimbre):
    bfcc_kmeans = zounds.ConstantRateTimeSeriesFeature(
            zounds.Learned,
            learned=BfccKmeans(),
            needs=WithTimbre.bfcc,
            store=True)

    sliding_bfcc_kmeans = zounds.ConstantRateTimeSeriesFeature(
            zounds.SlidingWindow,
            needs=bfcc_kmeans,
            wscheme=windowing * zounds.Stride(frequency=30, duration=30),
            store=False)

    bfcc_kmeans_pooled = zounds.ConstantRateTimeSeriesFeature(
            zounds.Max,
            needs=sliding_bfcc_kmeans,
            axis=1,
            store=True)


BaseIndex = zounds.hamming_index(WithCodes, WithCodes.bfcc_kmeans_pooled)


@zounds.simple_settings
class BfccKmeansIndex(BaseIndex):
    pass


if __name__ == '__main__':

    # Download the zip archive
    url = 'https://archive.org/download/FlavioGaete/FlavioGaete22.zip'
    filename = os.path.split(urlparse(url).path)[-1]

    resp = requests.get(url, stream=True)

    print 'Downloading {url} -> {filename}...'.format(**locals())

    with open(filename, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=4096):
            f.write(chunk)

    # stream all the audio files from the zip archive
    print 'Processing Audio...'
    for zf in ff.iter_zip(filename):
        if '._' in zf.filename:
            continue
        print zf.filename
        WithTimbre.process(meta=zf)

    # learn k-means codes for the bfcc frames
    print 'Learning K-Means clusters...'
    BfccKmeans.process(docs=(wt.bfcc for wt in WithTimbre))

    # build an index
    print 'Building index...'
    BfccKmeansIndex.build()
    index = BfccKmeansIndex()
    results = index.random_search()

    app = zounds.ZoundsApp(
            model=WithTimbre,
            audio_feature=WithTimbre.ogg,
            visualization_feature=WithTimbre.bark,
            globals=globals(),
            locals=locals())
    app.start(8888)
