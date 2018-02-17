import zounds
import numpy as np

samplerate = zounds.SR22050()
BaseModel = zounds.stft(resample_to=samplerate, store_fft=True)

band = zounds.FrequencyBand(50, samplerate.nyquist)
window = zounds.HanningWindowingFunc()


@zounds.simple_in_memory_settings
class Sound(BaseModel):
    chroma = zounds.ArrayWithUnitsFeature(
        zounds.Chroma,
        frequency_band=band,
        window=window,
        needs=BaseModel.fft)


if __name__ == '__main__':
    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.chroma,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals())

    port = 9999

    with app.start_in_thread(port):
        url = 'https://ia802606.us.archive.org/9/items/AOC11B/onclassical_luisi_bach_partita_B-flat-major_bwv-825_6.ogg'
        _id = Sound.process(meta=url)
        snd = Sound(_id)

        chroma_scale = zounds.ChromaScale(band)

        chroma = chroma_scale.apply(
            np.abs(snd.fft) * zounds.AWeighting(), window)

        basis = chroma_scale._basis(snd.fft.dimensions[-1].scale, window)

    app.start(port)
