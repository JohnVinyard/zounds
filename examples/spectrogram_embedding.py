"""
Use a triplet-loss to learn a similarity metric between short spectrograms

UNSUPERVISED LEARNING OF SEMANTIC AUDIO REPRESENTATIONS
https://arxiv.org/pdf/1711.02209.pdf
"""

import numpy as np
import zounds
from zounds.spectral import apply_scale

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

scale_bands = 96
spectrogram_duration = 64

anchor_slice = slice(spectrogram_duration, spectrogram_duration * 2)

scale = zounds.GeometricScale(
    start_center_hz=50,
    stop_center_hz=samplerate.nyquist,
    bandwidth_ratio=0.115,
    n_bands=scale_bands)
scale.ensure_overlap_ratio()

spectrogram_duration = 64

windowing_scheme = zounds.HalfLapped()
spectrogram_sample_rate = zounds.SampleRate(
    frequency=windowing_scheme.frequency * (spectrogram_duration // 2),
    duration=windowing_scheme.frequency * spectrogram_duration)


def spectrogram(x):
    x = apply_scale(
        np.abs(x.real), scale, window=zounds.OggVorbisWindowingFunc())
    x = zounds.log_modulus(x * 100)
    return x * zounds.AWeighting()


@zounds.simple_lmdb_settings(
    'spectrogram_embedding', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):
    short_windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=windowing_scheme,
        wfunc=zounds.OggVorbisWindowingFunc(),
        needs=BaseModel.resampled)

    fft = zounds.ArrayWithUnitsFeature(
        zounds.FFT,
        padding_samples=1024,
        needs=short_windowed)

    geom = zounds.ArrayWithUnitsFeature(
        spectrogram,
        needs=fft)

    log_spectrogram = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=zounds.SampleRate(
            frequency=windowing_scheme.frequency * (spectrogram_duration // 2),
            duration=windowing_scheme.frequency * spectrogram_duration * 3),
        needs=geom)

    ls = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=spectrogram_sample_rate,
        needs=geom)

if __name__ == '__main__':
    _id = Sound.process(
        meta='https://ia802606.us.archive.org/9/items/AOC11B/onclassical_luisi_bach_partita_B-flat-major_bwv-825_1.ogg')

    snd = Sound(_id)

    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.geom,
        audio_feature=Sound.ogg,
        globals=globals(),
        locals=locals())
    app.start(9000)
