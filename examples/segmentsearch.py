import featureflow as ff
import zounds
import numpy as np

windowing = zounds.HalfLapped()


# Segment audio files #########################################################

class Settings(ff.PersistenceSettings):
    id_provider = ff.UserSpecifiedIdProvider(key='_id')
    key_builder = ff.StringDelimitedKeyBuilder(seperator='|')
    database = ff.LmdbDatabase(path='onsetdata', key_builder=key_builder)
    event_log = ff.EventLog(
        path='onsetdataevents', channel=ff.InMemoryChannel())


STFT = zounds.stft(
    resample_to=zounds.SR11025(),
    wscheme=windowing)


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


# Store the K-Means encoding of Bark Bands ##############################

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


if __name__ == '__main__':
    index = zounds.HammingIndex(WithCodes, WithCodes.pooled, listen=True)

    # process the audio
    for request in zounds.PhatDrumLoops():
        try:
            WithOnsets.process(
                meta=request, _id=request.url, raise_if_exists=True)
            print 'processing {request.url}'.format(**locals())
        except Exception as e:
            print e

    # learn K-Means centroids
    try:
        print 'learning K-Means centroids'
        BarkKmeans.process(
            docs=(wo.bark for wo in WithOnsets),
            raise_if_exists=True)
    except ValueError:
        pass
    bark_kmeans = BarkKmeans()

    # force the new pooled feature to be computed
    for doc in WithCodes:
        print doc.pooled.slicedata.shape

    results = index.random_search(n_results=5)

    app = zounds.ZoundsSearch(
        model=WithCodes,
        audio_feature=WithCodes.ogg,
        visualization_feature=WithCodes.bark,
        search=index,
        n_results=5)

    app.start(8888)
