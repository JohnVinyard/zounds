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
    bark = zounds.ConstantRateTimeSeriesFeature(
            zounds.BarkBands,
            needs=STFT.fft,
            store=True)

    transience = zounds.ConstantRateTimeSeriesFeature(
            zounds.MeasureOfTransience,
            needs=STFT.fft,
            store=True)

    sliding_detection = zounds.ConstantRateTimeSeriesFeature(
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


def learn_kmeans():
    BarkKmeans.process(
            docs=(WithOnsets(_id).bark for _id in Settings.database))


# Store the K-Means representation of Bark Bands ##############################
class WithCodes(WithOnsets):
    bark_kmeans = zounds.ConstantRateTimeSeriesFeature(
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


# Create an index over the K-Keans codes for the onsets

def iter_codes():
    """
    Iterate over every summarized slice in the database
    """
    for _id in Settings.database:
        wc = WithCodes(_id)
        yield _id, wc.pooled.slicedata


@zounds.simple_settings
class BarkKmeansIndex(ff.BaseModel):
    codes = ff.Feature(
            ff.IteratorNode,
            store=False)

    contiguous = ff.NumpyFeature(
            zounds.Contiguous,
            needs=codes,
            encoder=ff.PackedNumpyEncoder,
            store=True)

    offsets = ff.PickleFeature(
            zounds.Offsets,
            needs=codes,
            store=True)


def build_kmeans_index():
    BarkKmeansIndex.process(codes=iter_codes())


def get_results(index):
    query_index = np.random.randint(0, len(index.contiguous))
    query = index.contiguous[query_index]
    return search.search(query)


if __name__ == '__main__':
    print 'getting audio...'
    get_audio()
    print 'learning k-means...'
    learn_kmeans()
    print 'building index...'
    build_kmeans_index()

    index = BarkKmeansIndex()

    search = zounds.Search(
            index,
            scorer=zounds.PackedHammingDistanceScorer(index),
            time_slice_builder=zounds.VariableRateTimeSliceBuilder(
                    index,
                    lambda x: WithCodes(x).pooled.slices))

    results = get_results(index)

    _ids = list(Settings.database.iter_ids())
    app = zounds.ZoundsApp(
            model=WithOnsets,
            audio_feature=WithOnsets.ogg,
            visualization_feature=WithOnsets.bark,
            globals=globals(),
            locals=locals())
    app.start(8888)
