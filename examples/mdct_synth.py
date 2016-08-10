import featureflow as ff
import zounds
from random import choice

samplerate = zounds.SR11025()
BaseDocument = zounds.stft(resample_to=samplerate)


@zounds.simple_lmdb_settings('mdct_synth', map_size=1e10)
class Document(BaseDocument):
    mdct = zounds.TimeFrequencyRepresentationFeature(
            zounds.MDCT,
            needs=BaseDocument.windowed,
            store=True)

    bark = zounds.ConstantRateTimeSeriesFeature(
            zounds.BarkBands,
            needs=BaseDocument.fft,
            store=True)


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

    unit_norm = ff.PickleFeature(
            zounds.UnitNorm,
            needs=log,
            store=False)

    kmeans = ff.PickleFeature(
            zounds.KMeans,
            centroids=512,
            needs=unit_norm,
            store=False)

    pipeline = ff.PickleFeature(
            zounds.PreprocessingPipeline,
            needs=(log, unit_norm, kmeans),
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
