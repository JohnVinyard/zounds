"""
Zounds implementation of:

A QUASI-ORTHOGONAL, INVERTIBLE, AND PERCEPTUALLY RELEVANT TIME-FREQUENCY
TRANSFORM FOR AUDIO CODING

http://www.eurasip.org/Proceedings/Eusipco/Eusipco2015/papers/1570092829.pdf
"""

from __future__ import division
import zounds
import featureflow as ff
import numpy as np
import scipy


class ErbMdctSynth(object):
    def __init__(self, linear_scale, log_scale, samplerate, windowing_func):
        super(ErbMdctSynth, self).__init__()
        self.windowing_func = windowing_func
        self.log_scale = log_scale
        self.linear_scale = linear_scale
        self.samplerate = zounds.audio_sample_rate(self.linear_scale.n_bands)

    def _idct(self, coeffs):
        return scipy.fftpack.idct(coeffs, norm='ortho')

    def synthesize(self, mdct_coeffs):

        frequency_dimension = zounds.FrequencyDimension(self.linear_scale)

        # initialize an empty array to fill with the dct coefficients
        dct_coeffs = zounds.ArrayWithUnits(
                np.zeros((len(mdct_coeffs), self.samplerate)),
                [mdct_coeffs.dimensions[0], frequency_dimension])

        weights = zounds.ArrayWithUnits(
                np.zeros(int(self.samplerate)), [frequency_dimension])
        for band in self.log_scale:
            weights[band] += 1
        weights[weights == 0] = 1
        weights = 1. / weights

        pos = 0
        for band in self.log_scale:
            slce = dct_coeffs[:, band]
            size = slce.shape[1]
            slce[:] += self._idct(mdct_coeffs[:, pos: pos + size])
            pos += size

        dct_synth = zounds.DCTSynthesizer(
                 windowing_func=self.windowing_func)
        return dct_synth.synthesize(dct_coeffs * weights)


class MDCT(ff.Node):
    def __init__(self, scale=None, windowing_func=None, needs=None):
        super(MDCT, self).__init__(needs=needs)
        self.windowing_func = windowing_func
        self.scale = scale

    def _mdct(self, data):
        # data = data * self.windowing_func
        return scipy.fftpack.dct(data, norm='ortho')

    def _process(self, data):
        transformed = np.concatenate([
            self._mdct(data[:, fb])
            for fb in self.scale], axis=1)

        yield zounds.ArrayWithUnits(
                transformed, [data.dimensions[0], zounds.IdentityDimension()])


samplerate = zounds.SR22050()
BaseModel = zounds.stft(resample_to=samplerate)

windowing_func = zounds.OggVorbisWindowingFunc()


scale = zounds.LogScale(
    zounds.FrequencyBand(1, 10000), n_bands=64)


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
                    zounds.Milliseconds(500),
                    zounds.Seconds(1)),
            wfunc=windowing_func,
            needs=BaseModel.resampled,
            store=True)

    dct = zounds.ArrayWithUnitsFeature(
            zounds.DCT,
            scale_always_even=True,
            needs=long_windowed,
            store=True)

    mdct = zounds.ArrayWithUnitsFeature(
            MDCT,
            scale=scale,
            windowing_func=windowing_func,
            needs=dct,
            store=True)


if __name__ == '__main__':

    # generate some audio
    synth = zounds.SineSynthesizer(zounds.SR22050())
    orig_audio = synth.synthesize(zounds.Seconds(5), [440., 660., 880.])

    # analyze the audio
    _id = Document.process(meta='http://www.phatdrumloops.com/audio/wav/lovedrops.wav')
    doc = Document(_id)

    # ensure that the dct-iv reconstruction is near-perfect
    dct_recon = zounds\
        .DCTSynthesizer(windowing_func=windowing_func)\
        .synthesize(doc.dct)

    # invert the representation
    synth = ErbMdctSynth(
            doc.dct.dimensions[1].scale,
            scale,
            samplerate,
            windowing_func)

    recon_audio = synth.synthesize(doc.mdct)

    app = zounds.ZoundsApp(
            model=Document,
            audio_feature=Document.ogg,
            visualization_feature=Document.bark,
            globals=globals(),
            locals=locals())
    app.start(8888)

