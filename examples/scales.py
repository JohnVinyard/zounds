import zounds

samplerate = zounds.SR22050()
BaseModel = zounds.stft(resample_to=samplerate, store_fft=True)


@zounds.simple_in_memory_settings
class Sound(BaseModel):
    pass


if __name__ == '__main__':
    url = 'https://ia802606.us.archive.org/9/items/AOC11B/onclassical_luisi_bach_partita_e-minor_bwv-830_3.ogg'
    _id = Sound.process(meta=url)
    snd = Sound(_id)

    band = zounds.FrequencyBand(50, samplerate.nyquist)
    bark_scale = zounds.BarkScale(band, 100)
    mel_scale = zounds.MelScale(band, 100)
    chroma_scale = zounds.ChromaScale(band)

    bark_bands = bark_scale.apply(snd.fft, zounds.HanningWindowingFunc())
    mel_bands = mel_scale.apply(snd.fft, zounds.HanningWindowingFunc())
    chroma_bands = chroma_scale.apply(snd.fft, zounds.HanningWindowingFunc())

    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.fft,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals())
    app.start(9999)