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


class ErbMdctSynth(object):

    def __init__(self, linear_scale, log_scale, samplerate, windowing_func):
        super(ErbMdctSynth, self).__init__()
        self.windowing_func = windowing_func
        self.log_scale = log_scale
        self.linear_scale = linear_scale
        self.samplerate = zounds.audio_sample_rate(self.linear_scale.n_bands)

    def _imdct(self, coeffs):
        l = coeffs.shape[1]
        t = np.arange(0, 2 * l)
        f = np.arange(0, l)
        cpi = -1j * np.pi
        a = coeffs * np.exp(cpi * (f + 0.5) * (l + 1) / 2 / l)
        b = np.fft.fft(a, 2 * l)
        transformed = np.sqrt(2 / l) * np.real(b * np.exp(cpi * t / 2 / l))
        return transformed * self.windowing_func

    def synthesize(self, mdct_coeffs):

        # initialize an empty array to fill with the dct coefficients
        dct_coeffs = zounds.ArrayWithUnits(
            np.zeros((len(mdct_coeffs), self.samplerate)), [
                mdct_coeffs.dimensions[0],
                zounds.FrequencyDimension(self.linear_scale)])

        # perform the inverse mdct on each frequncy band and populate the
        # dct coefficients
        pos = 0
        for band in self.log_scale:
            slce = dct_coeffs[:, band]
            mdct_size = slce.shape[1] // 2
            dct_coeffs[:, band] += \
                self._imdct(mdct_coeffs[:, pos: pos + mdct_size])
            pos += mdct_size

        dct_synth = zounds.DCTSynthesizer(
                windowing_func=self.windowing_func)
        return dct_synth.synthesize(dct_coeffs)


class MDCT(ff.Node):
    def __init__(self, scale=None, windowing_func=None, needs=None):
        super(MDCT, self).__init__(needs=needs)
        self.windowing_func = windowing_func
        self.scale = scale

    def _mdct(self, data):
        data *= self.windowing_func
        # KLUDGE: This is copied from the MDCT node in zounds.  Where does this
        # belong, and how do I make it re-usable?
        l = data.shape[1] // 2
        t = np.arange(0, 2 * l)
        f = np.arange(0, l)
        cpi = -1j * np.pi
        a = data * np.exp(cpi * t / 2 / l)
        b = np.fft.fft(a)
        c = b[:, :l]
        transformed = \
            np.sqrt(2 / l) * \
            np.real(c * np.exp(cpi * (f + 0.5) * (l + 1) / 2 / l))
        return transformed

    def _process(self, data):
        transformed = np.concatenate([
            self._mdct(data[:, fb])
            for fb in self.scale], axis=1)

        yield zounds.ArrayWithUnits(
                transformed, [data.dimensions[0], zounds.IdentityDimension()])


samplerate = zounds.SR22050()
BaseModel = zounds.stft(resample_to=samplerate)

windowing_func = zounds.OggVorbisWindowingFunc()

log_scale = zounds.LogScale(
        zounds.FrequencyBand(1, samplerate.nyquist),
        n_bands=128,
        always_even=True)


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
            scale=log_scale,
            windowing_func=windowing_func,
            needs=dct,
            store=True)


if __name__ == '__main__':

    # generate some audio
    synth = zounds.SineSynthesizer(zounds.SR22050())
    orig_audio = synth.synthesize(zounds.Seconds(10), [440., 660., 880.])

    # analyze the audio
    _id = Document.process(meta=orig_audio.encode())
    doc = Document(_id)

    # invert the long-windowed dct as a sanity check.  If this isn't a
    # near-perfect reconstruction, then things are already off-track
    dct_synth = zounds.DCTSynthesizer(windowing_func=windowing_func)
    dct_recon = dct_synth.synthesize(doc.dct)

    # invert the representation
    synth = ErbMdctSynth(
            doc.dct.dimensions[1].scale,
            log_scale,
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


