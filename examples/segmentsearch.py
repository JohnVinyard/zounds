import featureflow as ff
import zounds
import numpy as np
import requests
import re
import urlparse

windowing = zounds.HalfLapped()


# Segment audio files #########################################################

class Settings(ff.PersistenceSettings):
    id_provider = ff.UserSpecifiedIdProvider(key='_id')
    key_builder = ff.StringDelimitedKeyBuilder(seperator='|')
    database = ff.LmdbDatabase(path='onsetdata', key_builder=key_builder)


STFT = zounds.stft(resample_to=zounds.SR44100(), wscheme=windowing)


class WithOnsets(STFT, Settings):
    bark = zounds.ArrayWithUnitsFeature(
            zounds.BarkBands,
            needs=STFT.fft,
            store=True)

    transience = zounds.ArrayWithUnitsFeature(
            zounds.MeasureOfTransience,
            needs=STFT.fft,
            store=True)

    sliding_detection = zounds.ArrayWithUnitsFeature(
            zounds.SlidingWindow,
            needs=transience,
            wscheme=windowing * zounds.Stride(frequency=1, duration=11),
            padwith=5,
            store=False)

    slices = zounds.TimeSliceFeature(
            zounds.MovingAveragePeakPicker,
            needs=sliding_detection,
            aggregate=np.median,
            store=True)


# Learn K-Means Clusters ######################################################

@zounds.simple_settings
class BarkKmeans(ff.BaseModel):
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


# Store the K-Means representation of Bark Bands ##############################

class WithCodes(WithOnsets):
    bark_kmeans = zounds.ArrayWithUnitsFeature(
            zounds.Learned,
            learned=BarkKmeans(),
            needs=WithOnsets.bark,
            store=True)

    pooled = zounds.VariableRateTimeSeriesFeature(
            zounds.Pooled,
            needs=(bark_kmeans, WithOnsets.slices),
            op=np.max,
            axis=0,
            store=True)


def get_audio():
    resp = requests.get('http://phatdrumloops.com/beats.php')
    pattern = re.compile('href="(?P<uri>/audio/wav/[^\.]+\.wav)"')
    for m in pattern.finditer(resp.content):
        url = urlparse.urljoin('http://phatdrumloops.com', m.groupdict()['uri'])
        try:
            req = requests.Request(
                    method='GET',
                    url=url,
                    headers={'Range': 'bytes=0-'})
            WithOnsets.process(meta=req, _id=url)
            print 'Downloaded', url
        except Exception as e:
            print e
            pass


# Create an index over the K-Keans codes for the onsets #######################

@zounds.simple_settings
class BarkKmeansIndex(zounds.hamming_index(WithCodes, WithCodes.pooled)):
    pass


if __name__ == '__main__':
    print 'getting audio...'
    get_audio()
    print 'learning k-means...'
    BarkKmeans.process(docs=(wo.bark for wo in WithOnsets))
    print 'building index...'
    BarkKmeansIndex.build()

    index = BarkKmeansIndex()
    results = index.random_search()

    _ids = list(Settings.database.iter_ids())
    app = zounds.ZoundsApp(
            model=WithOnsets,
            audio_feature=WithOnsets.ogg,
            visualization_feature=WithOnsets.bark,
            globals=globals(),
            locals=locals())
    app.start(8888)
