"""
Demonstrate how to build a hamming-distance index over a binary/bit-packed
feature.

This example is particularly handy for performance profiling of the hamming
index.
"""

import zounds
import numpy as np

samplerate = zounds.SR11025()
BaseModel = zounds.stft(resample_to=samplerate, store_resampled=False)


def produce_fake_hash(x):
    """
    Produce random, binary features, totally irrespective of the content of
    x, but in the same shape as x.
    """
    h = np.random.binomial(1, 0.5, (x.shape[0], 1024))
    packed = np.packbits(h, axis=-1).view(np.uint64)
    return zounds.ArrayWithUnits(
        packed, [x.dimensions[0], zounds.IdentityDimension()])


@zounds.simple_lmdb_settings(
    'hamming_index', map_size=1e11, user_supplied_id=True)
class Sound(BaseModel):

    fake_hash = zounds.ArrayWithUnitsFeature(
        produce_fake_hash,
        needs=BaseModel.fft,
        store=True)


if __name__ == '__main__':

    zounds.ingest(
        zounds.InternetArchive('Kevin_Gates_-_By_Any_Means-2014'),
        Sound,
        multi_threaded=True)

    def web_url(doc, ts):
        return doc.meta['web_url']

    def total_duration(doc, ts):
        return doc.fake_hash.dimensions[0].end / zounds.Seconds(1)

    index = zounds.HammingIndex(
        Sound,
        Sound.fake_hash,
        path='fake_hash_index',
        web_url=web_url,
        total_duration=total_duration)

    if not len(index):
        index.add_all()

    for i in range(1000):
        list(index.random_search(n_results=50, sort=True))


