"""
Zounds implementation of:

A QUASI-ORTHOGONAL, INVERTIBLE, AND PERCEPTUALLY RELEVANT TIME-FREQUENCY
TRANSFORM FOR AUDIO CODING

http://www.eurasip.org/Proceedings/Eusipco/Eusipco2015/papers/1570092829.pdf
"""

import zounds
import featureflow as ff
import numpy as np


class MDCT(ff.Node):
    def __init__(self, n_bands=None, needs=None):
        super(MDCT, self).__init__(needs=needs)
        self.n_bands = n_bands

    def _first_chunk(self, data):
        duration = data.dimensions[0].duration_in_seconds
        # this assumes a DCT-IV transform, or some other transform where
        # the number of coefficients is equal to the number of samples
        samples_per_second = int(np.round(data.shape[1] / duration))
        self.samplerate = zounds.audio_sample_rate(samples_per_second)
        band = zounds.FrequencyBand(20, self.samplerate.nyquist)
        self.scale = zounds.LogScale(band, self.n_bands)
        return data

    def _mdct(self, data):
        # KLUDGE: This is copied from the MDCT node in zounds.  Where does this
        # belong, and how do I make it re-usable?
        l = data.shape[1] // 2
        t = np.arange(0, 2 * l)
        f = np.arange(0, l)
        cpi = -1j * np.pi
        a = data * np.exp(cpi * t / 2 / l)
        b = np.fft.fft(a)
        c = b[:, :l]
        transformed = np.sqrt(2 / l) * np.real(
                c * np.exp(cpi * (f + 0.5) * (l + 1) / 2 / l))
        return transformed

    def _process(self, data):
        transformed = np.concatenate([
             self._mdct(data[:, fb]) * zounds.OggVorbisWindowingFunc()
             for fb in self.scale], axis=1)
        yield zounds.ArrayWithUnits(
                transformed, [data.dimensions[0], zounds.IdentityDimension()])


BaseModel = zounds.resampled(resample_to=zounds.SR22050())


@zounds.simple_in_memory_settings
class Document(BaseModel):

    windowed = zounds.ArrayWithUnitsFeature(
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
            needs=windowed,
            store=True)

    mdct = zounds.ArrayWithUnitsFeature(
            MDCT,
            n_bands=64,
            needs=dctiv,
            store=True)


if __name__ == '__main__':

    # generate some audio
    synth = zounds.SineSynthesizer(zounds.SR22050())
    audio = synth.synthesize(zounds.Seconds(60), [440., 660., 880.])

    # analyze the audio
    _id = Document.process(meta=audio.encode())
    doc = Document(_id)
    print doc.mdct.shape

