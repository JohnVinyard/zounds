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
import featureflow as ff
import numpy as np
from scipy.fftpack import dct, idct


class Synth(object):
    """
    Invert the two-stage transformation by:
        1) applying an inverse DCT transform to the variable-sized frequency
           windows
        2) applying an inverse DCT transform to the fixed size time-frequency
           representation
    """

    def __init__(self, linear_scale, log_scale, samplerate, windowing_func):
        super(Synth, self).__init__()
        self.windowing_func = windowing_func
        self.log_scale = log_scale
        self.linear_scale = linear_scale
        self.samplerate = zounds.audio_sample_rate(self.linear_scale.n_bands)

    def _weights(self, frequency_dimension):
        """
        Compute weights to compensate for the fact that overlapping windows
        may over-emphasize or de-emphasize frequencies at certain points
        :param frequency_dimension: The frequency scale onto which these weights
        map
        :return: computed weights
        """
        weights = zounds.ArrayWithUnits(
                np.zeros(int(self.samplerate)), [frequency_dimension])
        for band in self.log_scale:
            weights[band] += 1
        weights[weights == 0] = 1
        return 1. / weights

    def synthesize(self, mdct_coeffs):

        frequency_dimension = zounds.FrequencyDimension(self.linear_scale)

        # initialize an empty array to fill with the dct coefficients
        dct_coeffs = zounds.ArrayWithUnits(
                np.zeros((len(mdct_coeffs), self.samplerate)),
                [mdct_coeffs.dimensions[0], frequency_dimension])

        # compute compensating weights
        weights = self._weights(frequency_dimension)

        # invert the variable-size frequency windows
        pos = 0
        for band in self.log_scale:
            slce = dct_coeffs[:, band]
            size = slce.shape[1]
            slce[:] += idct(mdct_coeffs[:, pos: pos + size], norm='ortho')
            pos += size

        # invert the fixed-size time-frequency representation
        dct_synth = zounds.DCTSynthesizer(
                windowing_func=self.windowing_func)
        return dct_synth.synthesize(dct_coeffs * weights)


class VariableSizedFrequencyWindows(ff.Node):
    """
    Given a fixed-size time-frequency representation, compute DCT coefficients
    over variable-sized frequency windows that follow a logarithmic scale, which
    maps more closely onto the critical bands of hearing
    """
    def __init__(self, scale=None, windowing_func=None, needs=None):
        super(VariableSizedFrequencyWindows, self).__init__(needs=needs)
        self.windowing_func = windowing_func
        self.scale = scale

    def _process(self, data):
        transformed = np.concatenate([
            dct(data[:, fb], norm='ortho')
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
            VariableSizedFrequencyWindows,
            scale=scale,
            windowing_func=windowing_func,
            needs=dct,
            store=True)


if __name__ == '__main__':
    # generate some audio
    synth = zounds.SineSynthesizer(zounds.SR22050())
    orig_audio = synth.synthesize(zounds.Seconds(5), [440., 660., 880.])

    # analyze the audio
    _id = Document.process(meta=orig_audio.encode())
    doc = Document(_id)

    # invert the representation
    synth = Synth(
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
