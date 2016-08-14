import featureflow as ff
import zounds
from random import choice

samplerate = zounds.SR11025()
BaseDocument = zounds.stft(resample_to=samplerate)


@zounds.simple_lmdb_settings('mdct_synth', map_size=1e10)
class Document(BaseDocument):

    # compute the MDCT over a sliding window of time-domain samples
    mdct = zounds.TimeFrequencyRepresentationFeature(
            zounds.MDCT,
            needs=BaseDocument.windowed,
            store=True)

    # compute bark bands, for display purposes
    bark = zounds.ConstantRateTimeSeriesFeature(
            zounds.BarkBands,
            needs=BaseDocument.fft,
            store=True)

# compute the array needed to perceptually weight the MDCT frequencies
scale = zounds.LinearScale.from_sample_rate(samplerate, 256)
weights = zounds.AWeighting().weights(scale)


@zounds.simple_settings
class DctKmeans(ff.BaseModel):
    docs = ff.Feature(
            ff.IteratorNode,
            store=False)

    shuffle = ff.NumpyFeature(
            zounds.ReservoirSampler,
            nsamples=1e6,
            needs=docs,
            store=True)

    log = ff.PickleFeature(
            zounds.Log,
            needs=shuffle,
            store=False)

    # apply the perceptual weighting
    # TOOD: Try this before and after log amplitude
    multiply = ff.PickleFeature(
            zounds.Multiply,
            factor=weights,
            needs=log,
            store=False)

    unit_norm = ff.PickleFeature(
            zounds.UnitNorm,
            needs=multiply,
            store=False)

    kmeans = ff.PickleFeature(
            zounds.KMeans,
            centroids=512,
            needs=unit_norm,
            store=False)

    pipeline = ff.PickleFeature(
            zounds.PreprocessingPipeline,
            needs=(log, multiply, unit_norm, kmeans),
            store=True)


@zounds.simple_lmdb_settings('mdct_synth_with_codes', map_size=1e10)
class WithCodes(Document):
    kmeans = zounds.ConstantRateTimeSeriesFeature(
            zounds.Learned,
            learned=DctKmeans(),
            needs=Document.mdct,
            store=True)


if __name__ == '__main__':

    # stream all the audio files from the zip archive
    filename = 'FlavioGaete22.zip'
    print 'Processing Audio...'
    for zf in ff.iter_zip(filename):
        if '._' in zf.filename:
            continue
        print zf.filename
        Document.process(meta=zf)

    # learn k-means clusters for the mdct frames
    print 'learning k-means clusters'
    DctKmeans.process(docs=(doc.mdct for doc in Document))

    synth = zounds.MDCTSynthesizer()
    docs = list(doc for doc in WithCodes)
    kmeans = DctKmeans()


    def random_reconstruction():
        doc = choice(docs)
        transform_result = kmeans.pipeline.transform(doc.mdct)
        recon_mdct = transform_result.inverse_transform()
        recon_audio = synth.synthesize(recon_mdct)
        return doc.ogg, recon_audio


    app = zounds.ZoundsApp(
            model=Document,
            audio_feature=Document.ogg,
            visualization_feature=Document.bark,
            globals=globals(),
            locals=locals())
    app.start(8888)
