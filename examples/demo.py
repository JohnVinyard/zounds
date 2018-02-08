import zounds

Resampled = zounds.resampled(resample_to=zounds.SR11025())


@zounds.simple_in_memory_settings
class Sound(Resampled):
    """
    A simple pipeline that computes a perceptually weighted modified discrete
    cosine transform, and "persists" feature data in an in-memory store.
    """

    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        needs=Resampled.resampled,
        wscheme=zounds.HalfLapped(),
        wfunc=zounds.OggVorbisWindowingFunc(),
        store=True)

    mdct = zounds.ArrayWithUnitsFeature(
        zounds.MDCT,
        needs=windowed)

    weighted = zounds.ArrayWithUnitsFeature(
        lambda x: x * zounds.AWeighting(),
        needs=mdct)

if __name__ == '__main__':

    # produce some audio to test our pipeline, and encode it as FLAC
    synth = zounds.SineSynthesizer(zounds.SR44100())
    samples = synth.synthesize(zounds.Seconds(5), [220., 440., 880.])
    encoded = samples.encode(fmt='FLAC')

    # process the audio, and fetch features from our in-memory store
    _id = Sound.process(meta=encoded)
    sound = Sound(_id)

    # grab all the frequency information, for a subset of the duration
    start = zounds.Milliseconds(500)
    end = start + zounds.Seconds(2)
    snippet = sound.weighted[start: end, :]

    # grab a subset of frequency information for the duration of the sound
    freq_band = slice(zounds.Hertz(400), zounds.Hertz(500))
    a440 = sound.mdct[:, freq_band]

    # produce a new set of coefficients where only the 440hz sine wave is
    # present
    filtered = sound.mdct.zeros_like()
    filtered[:, freq_band] = a440

    # apply a geometric scale, which more closely matches human pitch
    # perception, and apply it to the linear frequency axis
    scale = zounds.GeometricScale(50, 4000, 0.05, 100)
    log_coeffs = scale.apply(sound.mdct, zounds.HanningWindowingFunc())

    # reconstruct audio from the MDCT coefficients
    mdct_synth = zounds.MDCTSynthesizer()
    reconstructed = mdct_synth.synthesize(sound.mdct)
    filtered_reconstruction = mdct_synth.synthesize(filtered)

    # start an in-browser REPL that will allow you to listen to and visualize
    # the variables defined above (and any new ones you create in the session)
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.weighted,
        globals=globals(),
        locals=locals())
    app.start(9999)
