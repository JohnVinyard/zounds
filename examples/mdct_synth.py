"""
Demonstrate an extremely simple audio encoder that learns basis functions for
individual audio frames from a corpus of data
"""

import featureflow as ff
import zounds
from random import choice

samplerate = zounds.SR11025()
STFT = zounds.stft(resample_to=samplerate)


class Settings(ff.PersistenceSettings):
    """
    These settings make it possible to specify an id (rather than automatically
    generating one) when analyzing a file, so that it's easier to reference
    them later.
    """
    id_provider = ff.UserSpecifiedIdProvider(key='_id')
    key_builder = ff.StringDelimitedKeyBuilder()
    database = ff.LmdbDatabase(
            'mdct_synth', map_size=1e10, key_builder=key_builder)


class Document(STFT, Settings):
    """
    Inherit from a basic processing graph, and add a Modified Discrete Cosine
    Transform feature
    """
    mdct = zounds.TimeFrequencyRepresentationFeature(
            zounds.MDCT,
            needs=STFT.windowed,
            store=True)


@zounds.simple_settings
class DctKmeans(ff.BaseModel):
    """
    A pipeline that does example-wise normalization by giving each example
    unit-norm, and learns 512 centroids from those examples.
    """
    docs = ff.Feature(
            ff.IteratorNode,
            store=False)

    # randomize the order of the data
    shuffle = ff.NumpyFeature(
            zounds.ReservoirSampler,
            nsamples=1e6,
            needs=docs,
            store=True)

    # give each frame unit norm, since we care about the shape of the spectrum
    # and not its magnitude
    unit_norm = ff.PickleFeature(
            zounds.UnitNorm,
            needs=shuffle,
            store=False)

    # learn 512 centroids, or basis functions
    kmeans = ff.PickleFeature(
            zounds.KMeans,
            centroids=512,
            needs=unit_norm,
            store=False)

    # assemble the previous steps into a re-usable pipeline, which can perform
    # forward and backward transformations
    pipeline = ff.PickleFeature(
            zounds.PreprocessingPipeline,
            needs=(unit_norm, kmeans),
            store=True)


@zounds.simple_settings
class DctKmeansWithLogAmplitude(ff.BaseModel):
    """
    A pipeline that applies a logarithmic weighting to the magnitudes of the
    spectrum before learning centroids,
    """
    docs = ff.Feature(
            ff.IteratorNode,
            store=False)

    # randomize the order of the data
    shuffle = ff.NumpyFeature(
            zounds.ReservoirSampler,
            nsamples=1e6,
            needs=docs,
            store=True)

    log = ff.PickleFeature(
            zounds.Log,
            needs=shuffle,
            store=False)

    # give each frame unit norm, since we care about the shape of the spectrum
    # and not its magnitude
    unit_norm = ff.PickleFeature(
            zounds.UnitNorm,
            needs=log,
            store=False)

    # learn 512 centroids, or basis functions
    kmeans = ff.PickleFeature(
            zounds.KMeans,
            centroids=512,
            needs=unit_norm,
            store=False)

    # assemble the previous steps into a re-usable pipeline, which can perform
    # forward and backward transformations
    pipeline = ff.PickleFeature(
            zounds.PreprocessingPipeline,
            needs=(log, unit_norm, kmeans),
            store=True)


if __name__ == '__main__':

    # stream all the audio files from the zip archive
    # you can download the original file here:
    # https://archive.org/details/FlavioGaete
    # - https://archive.org/download/FlavioGaete/FlavioGaete22.zip
    filename = 'FlavioGaete22.zip'
    print 'Processing Audio...'
    for zf in ff.iter_zip(filename):
        if '._' in zf.filename:
            continue
        try:
            # check if the feature already exists
            Document(zf.filename).mdct.shape
        except KeyError:
            print 'processing {filename}'.format(filename=zf.filename)
            Document.process(meta=zf, _id=zf.filename)

    print 'learn k-means clusters'
    DctKmeans.process(docs=(doc.mdct for doc in Document))

    print 'learn k-means clusters with log amplitude'
    DctKmeansWithLogAmplitude.process(docs=(doc.mdct for doc in Document))

    synth = zounds.MDCTSynthesizer()
    docs = list(doc for doc in Document)
    kmeans = DctKmeans()
    kmeans_log_amplitude = DctKmeansWithLogAmplitude()


    def full_pass(mdct, pipeline):
        """
        Do a forward and backward pass over the audio, and return the
        reconstruction of the MDCT coefficients
        """
        transform_result = pipeline.transform(mdct)
        recon_mdct = transform_result.inverse_transform()
        recon_audio = synth.synthesize(recon_mdct)
        return recon_audio


    def reconstruction(_id=None):
        """
        Do a forward and backward pass over the audio, and return the
        reconstructed audio
        """
        if _id:
            doc = Document(_id)
        else:
            doc = choice(docs)
        print doc._id
        recon_audio = full_pass(doc.mdct, kmeans.pipeline)
        recon_audio_log_amp = full_pass(doc.mdct, kmeans_log_amplitude.pipeline)
        return doc.ogg[:], recon_audio, recon_audio_log_amp


    mono_orig, mono, mono_log = reconstruction(
            'FlavioGaete22/TFS1_TReich11.wav')
    bass_orig, bass, bass_log = reconstruction(
            'FlavioGaete22/TFS1_TBass05.wav')
    beat_orig, beat, beat_log = reconstruction(
            'FlavioGaete22/SIKBeat02.wav')
    cello_orig, cello, cello_log = reconstruction(
            'FlavioGaete22/TFS2_TVla09.wav')

    app = zounds.ZoundsApp(
            model=Document,
            audio_feature=Document.ogg,
            visualization_feature=Document.mdct,
            globals=globals(),
            locals=locals())
    app.start(8888)
