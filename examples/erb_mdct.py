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

    def __init__(self, linear_scale, log_scale, samplerate):
        super(ErbMdctSynth, self).__init__()
        self.log_scale = log_scale
        self.linear_scale = linear_scale
        print self.linear_scale
        self.samplerate = zounds.audio_sample_rate(self.linear_scale.n_bands)

    def _imdct(self, coeffs):
        l = coeffs.shape[1]
        t = np.arange(0, 2 * l)
        f = np.arange(0, l)
        cpi = -1j * np.pi
        a = coeffs * np.exp(cpi * (f + 0.5) * (l + 1) / 2 / l)
        b = np.fft.fft(a, 2 * l)
        transformed = np.sqrt(2 / l) * np.real(b * np.exp(cpi * t / 2 / l))
        return transformed * zounds.OggVorbisWindowingFunc()

    def synthesize(self, mdct_coeffs):

        # initialize an empty array to fill with the dct-iv coefficients
        dct_iv_coeffs = zounds.ArrayWithUnits(
            np.zeros((len(mdct_coeffs), self.samplerate)), [
                mdct_coeffs.dimensions[0],
                zounds.FrequencyDimension(self.linear_scale)])

        # perform the inverse mdct on each frequncy band and populate the
        # dct-iv coefficients
        pos = 0
        for band in self.log_scale:
            slce = dct_iv_coeffs[:, band]
            mdct_size = slce.shape[1] // 2
            dct_iv_coeffs[:, band] += \
                self._imdct(mdct_coeffs[:, pos: pos + mdct_size])
            pos += mdct_size

        dct_iv_synth = zounds.DCTIVSynthesizer()
        return dct_iv_coeffs, dct_iv_synth.synthesize(dct_iv_coeffs)


class MDCT(ff.Node):
    def __init__(self, scale=None, needs=None):
        super(MDCT, self).__init__(needs=needs)
        self.scale = scale

    def _mdct(self, data):
        data *= zounds.OggVorbisWindowingFunc()
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
            wfunc=zounds.OggVorbisWindowingFunc(),
            needs=BaseModel.resampled,
            store=True)

    dctiv = zounds.ArrayWithUnitsFeature(
            zounds.DCTIV,
            scale_always_even=True,
            needs=long_windowed,
            store=True)

    mdct = zounds.ArrayWithUnitsFeature(
            MDCT,
            scale=log_scale,
            needs=dctiv,
            store=True)


if __name__ == '__main__':

    # generate some audio
    synth = zounds.SineSynthesizer(zounds.SR22050())
    orig_audio = synth.synthesize(zounds.Seconds(10), [440., 660., 880.])

    # analyze the audio
    _id = Document.process(meta=orig_audio.encode())
    doc = Document(_id)

    # def check_cola(win_func):
    #     # look at the windows I'm using
    #     coeffs = doc.dctiv[0]
    #     empty = np.zeros(coeffs.shape)
    #
    #     freq_dim = coeffs.dimensions[0]
    #
    #     # TODO: How much overlap is there?  Is it enough to satisfy the needs
    #     # of the MDCT?
    #     bands = list(log_scale)
    #     for i in xrange(len(bands[:-1])):
    #         current = bands[i]
    #         next = bands[i + 1]
    #         current_slice = freq_dim.scale.get_slice(current)
    #         next_slice = freq_dim.scale.get_slice(next)
    #         current_size = current_slice.stop - current_slice.start
    #         half_current_size = current_size // 2
    #         latest_start_point = current_slice.start + half_current_size
    #         actual_start_point = next_slice.start
    #         print latest_start_point, actual_start_point
    #         if actual_start_point > latest_start_point:
    #             print 'FAIL'
    #
    #     # TODO: How can I find the perfect window?
    #     for band in log_scale:
    #         slce = freq_dim.scale.get_slice(band)
    #         size = slce.stop - slce.start
    #         empty[slce] += win_func(size)
    #
    #     # look at some "perfect" windows
    #     perfect = np.zeros(coeffs.shape)
    #     for i in xrange(0, len(perfect) - 2048, 1024):
    #         perfect[i: i + 2048] += win_func(2048)
    #
    #     return empty, perfect


    # invert the representation
    synth = ErbMdctSynth(
            doc.dctiv.dimensions[1].scale,
            log_scale,
            samplerate)

    dct_iv_coeffs, recon_audio = synth.synthesize(doc.mdct)

    app = zounds.ZoundsApp(
        model=Document,
        audio_feature=Document.ogg,
        visualization_feature=Document.bark,
        globals=globals(),
        locals=locals())
    app.start(8888)


