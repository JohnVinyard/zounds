import zounds
import argparse

samplerate = zounds.SR11025()
BaseModel = zounds.stft(resample_to=samplerate, store_fft=True)


@zounds.simple_in_memory_settings
class Sound(BaseModel):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local-path',
        required=True,
        type=str,
        help='local path where music net csv and npz files should be stored')
    parser.add_argument(
        '--port',
        default=8888,
        type=int,
        help='port to run the in-browser REPL in')
    args = parser.parse_args()

    args = parser.parse_args()

    mn = zounds.MusicNet(path=args.local_path)
    zounds.ingest(mn, Sound, multi_threaded=True)

    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.fft,
        globals=globals(),
        locals=locals())
    app.start(args.port)
