"""
Zounds implementation of something similar to/inspired by:

A QUASI-ORTHOGONAL, INVERTIBLE, AND PERCEPTUALLY RELEVANT TIME-FREQUENCY
TRANSFORM FOR AUDIO CODING

http://www.eurasip.org/Proceedings/Eusipco/Eusipco2015/papers/1570092829.pdf

This implementation differs in that it does not use the MDCT transform on the
frequency domain, as getting the overlaps just right, such that they satisfy
MDCT invertibility requirements, is tricky, and requires some low level
knowledge that zounds' Scale attempts to abstract away.

See section 3.3 Setting MDCT Sizes for information about what we're fudging/
glossing over in this implementation.  We instead use the DCT2 transform, which
makes inversion easier, at the cost of more redundancy.
"""

from __future__ import division
import zounds
import scipy

samplerate = zounds.SR11025()
BaseModel = zounds.stft(resample_to=samplerate)

windowing_func = zounds.OggVorbisWindowingFunc()

scale = zounds.GeometricScale(300, 3030, 0.05, 100)


@zounds.simple_in_memory_settings
class Document(BaseModel):
    bark = zounds.ArrayWithUnitsFeature(
        zounds.BarkBands,
        samplerate=samplerate,
        stop_freq_hz=samplerate.nyquist,
        needs=BaseModel.fft,
        store=True)

    long_windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=zounds.SampleRate(
            frequency=zounds.Milliseconds(500),
            duration=zounds.Seconds(1)),
        wfunc=windowing_func,
        needs=BaseModel.resampled,
        store=True)

    dct = zounds.ArrayWithUnitsFeature(
        zounds.DCT,
        scale_always_even=True,
        needs=long_windowed,
        store=True)

    mdct = zounds.FrequencyAdaptiveFeature(
        zounds.FrequencyAdaptiveTransform,
        transform=scipy.fftpack.idct,
        scale=scale,
        needs=dct,
        store=True)


if __name__ == '__main__':
    # generate some audio
    synth = zounds.TickSynthesizer(zounds.SR22050())
    orig_audio = synth.synthesize(zounds.Seconds(5), zounds.Milliseconds(200))

    # analyze the audio
    _id = Document.process(meta=orig_audio.encode())
    doc = Document(_id)

    synth = zounds.FrequencyAdaptiveDCTSynthesizer(scale, samplerate)
    recon_audio = synth.synthesize(doc.mdct)

    # get a rasterized visualization of the representation
    img = doc.mdct.square(100, do_overlap_add=True)

    app = zounds.ZoundsApp(
        model=Document,
        audio_feature=Document.ogg,
        visualization_feature=Document.bark,
        globals=globals(),
        locals=locals())
    app.start(8888)
