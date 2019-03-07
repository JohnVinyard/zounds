import os
from urllib.parse import urlparse
import featureflow as ff
import requests
import zounds


class Settings(ff.PersistenceSettings):
    id_provider = ff.UserSpecifiedIdProvider('_id')
    key_builder = ff.StringDelimitedKeyBuilder(seperator='|')
    database = ff.LmdbDatabase(path='timbre', key_builder=key_builder)
    event_log = ff.EventLog('timbre_events', channel=ff.InMemoryChannel())


windowing = zounds.HalfLapped()
STFT = zounds.stft(resample_to=zounds.SR22050(), wscheme=windowing)


class WithTimbre(STFT, Settings):
    bark = zounds.ArrayWithUnitsFeature(
        zounds.BarkBands,
        needs=STFT.fft,
        store=True)

    bfcc = zounds.ArrayWithUnitsFeature(
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
    bfcc_kmeans = zounds.ArrayWithUnitsFeature(
        zounds.Learned,
        learned=BfccKmeans(),
        needs=WithTimbre.bfcc,
        store=True)

    sliding_bfcc_kmeans = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        needs=bfcc_kmeans,
        wscheme=windowing * zounds.Stride(frequency=30, duration=30),
        store=False)

    bfcc_kmeans_pooled = zounds.ArrayWithUnitsFeature(
        zounds.Max,
        needs=sliding_bfcc_kmeans,
        axis=1,
        store=True)


def download_zip_archive():
    # Download the zip archive
    url = 'https://archive.org/download/FlavioGaete/FlavioGaete22.zip'
    filename = os.path.split(urlparse(url).path)[-1]

    if not os.path.exists(filename):
        resp = requests.get(url, stream=True)

        print('Downloading {url} -> {filename}...'.format(**locals()))

        with open(filename, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1000000):
                f.write(chunk)

    return filename

if __name__ == '__main__':
    index = zounds.HammingIndex(
        WithCodes, WithCodes.bfcc_kmeans_pooled, listen=True)

    zip_filename = download_zip_archive()

    print('Processing Audio...')
    for zf in ff.iter_zip(zip_filename):

        if '._' in zf.filename:
            continue

        try:
            print('processing {zf.filename}'.format(**locals()))
            WithTimbre.process(
                _id=zf.filename, meta=zf, raise_if_exists=True)
        except ff.ModelExistsError as e:
            print(e)

    # learn K-Means centroids
    try:
        print('learning K-Means centroids')
        BfccKmeans.process(
            docs=(wt.bfcc for wt in WithTimbre), raise_if_exists=True)
    except ff.ModelExistsError:
        pass

    # force the new features to be computed, so they're pushed into the index
    for wc in WithCodes:
        print(wc.bfcc_kmeans_pooled)

    app = zounds.ZoundsSearch(
        model=WithTimbre,
        audio_feature=WithTimbre.ogg,
        visualization_feature=WithTimbre.bark,
        search=index,
        n_results=10)
    app.start(8888)
