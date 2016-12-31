"""
Zounds implementation of

http://www.eurasip.org/Proceedings/Eusipco/Eusipco2015/papers/1570092829.pdf
"""

import zounds

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
            needs=windowed,
            store=True)


if __name__ == '__main__':
    synth = zounds.SineSynthesizer(zounds.SR22050())
    audio = synth.synthesize(zounds.Seconds(10), [440., 660., 880.])

    _id = Document.process(meta=audio.encode())
    doc = Document(_id)

    band = zounds.FrequencyBand(20, 11025)
    scale = zounds.LogScale(band, 128)

    for band in scale:
        slce = doc.dctiv[:, band]
        print slce.dimensions
        print slce.shape
        print band



