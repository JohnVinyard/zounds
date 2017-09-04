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
        needs=windowed,
        store=False)

    weighted = zounds.ArrayWithUnitsFeature(
        zounds.FrequencyWeighting,
        weighting=zounds.AWeighting(),
        needs=mdct,
        store=False)

if __name__ == '__main__':

    # produce some audio to test our pipeline
    synth = zounds.SineSynthesizer(zounds.SR44100())
    samples = synth.synthesize(zounds.Seconds(5), [220., 440., 880.])

    # process the audio, and fetch features from our in-memory store
    _id = Sound.process(meta=samples.encode())
    sound = Sound(_id)

    # produce a time slice that starts half a second in, and lasts for two
    # seconds
    time_slice = zounds.TimeSlice(
        start=zounds.Milliseconds(500),
        duration=zounds.Seconds(2))
    # grab all the frequency information, for a subset of the duration
    snippet = sound.weighted[time_slice, :]

    # produce a frequency slice that spans 400hz-500hz
    freq_band = zounds.FrequencyBand(400, 500)
    # grab a subset of frequency information for the duration of the sound
    a440 = sound.mdct[:, freq_band]

    # produce a new set of coefficients where only the 440hz sine wave is
    # present
    filtered = sound.mdct.copy()
    filtered[:] = 0
    filtered[:, freq_band] = a440

    # apply a geometric scale, which more closely matches human pitch
    # perception, and apply it to the linear frequency axis
    scale = zounds.GeometricScale(50, 4000, 0.05, 100)
    bands = [sound.weighted[:, band] for band in scale]
    band_sizes = [band.shape[1] for band in bands]

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
    app.start(8888)
