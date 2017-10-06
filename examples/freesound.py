"""
Demonstrate how to download and process sounds from https://freesound.org
"""

import zounds
import argparse

BaseModel = zounds.stft(resample_to=zounds.SR11025())


@zounds.simple_lmdb_settings('freesound', map_size=1e10, user_supplied_id=True)
class Sound(BaseModel):
    bark = zounds.ArrayWithUnitsFeature(
        zounds.BarkBands,
        needs=BaseModel.fft,
        store=True)

    chroma = zounds.ArrayWithUnitsFeature(
        zounds.Chroma,
        needs=BaseModel.fft,
        store=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api-key',
        help='your Freesound API key (http://freesound.org/apiv2/apply/)',
        type=str,
        required=True)
    parser.add_argument(
        '--query',
        help='the text query to run against freesound',
        type=str,
        required=True)
    args = parser.parse_args()

    fss = zounds.FreeSoundSearch(args.api_key, args.query)

    for metadata in fss:
        request = metadata.request
        url = request.url
        if not Sound.exists(url):
            Sound.process(meta=metadata, _id=url)
            print 'processed {url}'.format(**locals())
        else:
            print 'already processed {url}'.format(**locals())

    snds = list(Sound)

    # start an in-browser REPL that will allow you to listen to and visualize
    # the variables defined above (and any new ones you create in the session)
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.bark,
        globals=globals(),
        locals=locals())
    app.start(8888)
