#!/usr/bin/env python

import featureflow as ff
import zounds
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa('--datadir',
       help='the directory where data will be stored',
       required=True,
       type=str)
    aa('--port',
       help='the port number to run the server on',
       required=False,
       type=int,
       default=8888)
    aa('--freesoundkey',
       help='freesound.org api key',
       required=False,
       type=str)

    args = parser.parse_args()


    class Settings(ff.PersistenceSettings):
        id_provider = ff.UuidProvider()
        key_builder = ff.StringDelimitedKeyBuilder()
        database = ff.LmdbDatabase(
                path=args.datadir, key_builder=key_builder)

    AudioGraph = zounds.audio_graph()
    WithOnsets = zounds.with_onsets(AudioGraph.fft)

    class Document(AudioGraph, WithOnsets, Settings):
        pass

    if not os.path.exists(args.datadir):
        os.makedirs(args.datadir)

    if args.freesoundkey:
        freesound = zounds.FreesoundOrgConfig(args.freesoundkey)

    synth = zounds.DCTSynthesizer()

    app = zounds.ZoundsApp(
            model=Document,
            audio_feature=Document.ogg,
            visualization_feature=Document.bark,
            globals=globals(),
            locals=locals())
    app.start(args.port)
