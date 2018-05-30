import zounds
from zounds.spectral import time_stretch, pitch_shift
from zounds.ui import AppSettings
import argparse

sr = zounds.SR11025()
BaseModel = zounds.stft(resample_to=sr, store_resampled=True)


@zounds.simple_in_memory_settings
class Sound(BaseModel):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[
        AppSettings()
    ])
    parser.add_argument(
        '--sound-uri',
        default='https://archive.org/download/LucaBrasi2/06-Kevin_Gates-Out_The_Mud_Prod_By_The_Runners_The_Monarch.ogg')
    args = parser.parse_args()

    _id = Sound.process(meta=args.sound_uri)
    snd = Sound(_id)

    original = snd.resampled
    slow = zounds.AudioSamples(time_stretch(original, 0.75).squeeze(), sr)
    fast = zounds.AudioSamples(time_stretch(original, 1.25).squeeze(), sr)

    higher = zounds.AudioSamples(
        pitch_shift(original[:zounds.Seconds(10)], 1.0).squeeze(), sr)
    lower = zounds.AudioSamples(
        pitch_shift(original[:zounds.Seconds(10)], -1.0).squeeze(), sr)

    app = zounds.ZoundsApp(
        model=Sound,
        visualization_feature=Sound.fft,
        audio_feature=Sound.resampled,
        globals=globals(),
        locals=locals(),
        secret=args.app_secret)

    app.start(args.port)
