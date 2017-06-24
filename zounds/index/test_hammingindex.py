import unittest2
from index import HammingIndex
from featureflow import \
    PersistenceSettings, UuidProvider, StringDelimitedKeyBuilder, \
    InMemoryDatabase, InMemoryChannel, EventLog
from zounds.basic import stft, Slice, Binarize
from zounds.timeseries import SR11025, Seconds
from zounds.synthesize import SineSynthesizer
from zounds.persistence import ArrayWithUnitsFeature
import shutil
from uuid import uuid4


class HammingIndexTests(unittest2.TestCase):
    def setUp(self):
        self.event_log_path = '/tmp/{path}'.format(path=uuid4().hex)
        self.hamming_db_path = '/tmp/{path}'.format(path=uuid4().hex)

    def tearDown(self):
        shutil.rmtree(self.event_log_path, ignore_errors=True)
        shutil.rmtree(self.hamming_db_path, ignore_errors=True)

    def test_raises_if_model_class_has_no_event_log_configured(self):
        Model = self._model(
            slice_size=64,
            settings=self._settings_with_no_event_log())
        self.assertRaises(ValueError, lambda: HammingIndex(Model, Model.sliced))

    def test_correctly_infers_code_size_8(self):
        Model = self._model(
            slice_size=64,
            settings=self._settings_with_event_log())

        index = self._index(Model, Model.sliced)
        signal = SineSynthesizer(SR11025()) \
            .synthesize(Seconds(5), [220, 440, 880])
        Model.process(meta=signal.encode())
        index._synchronously_process_events()
        self.assertEqual(8, index.hamming_db.code_size)

    def test_correctly_infers_code_size_16(self):
        Model = self._model(
            slice_size=128,
            settings=self._settings_with_event_log())

        index = self._index(Model, Model.sliced)
        signal = SineSynthesizer(SR11025()) \
            .synthesize(Seconds(5), [220, 440, 880])
        Model.process(meta=signal.encode())
        index._synchronously_process_events()
        self.assertEqual(16, index.hamming_db.code_size)

    def correctly_infers_index_name(self):
        Model = self._model(
            slice_size=128,
            settings=self._settings_with_event_log())

        index = self._index(Model, Model.sliced)
        signal = SineSynthesizer(SR11025()) \
            .synthesize(Seconds(5), [220, 440, 880])
        Model.process(meta=signal.encode())
        index._synchronously_process_events()
        self.assertTrue('index.sliced' in index.hamming_db.path)

    def _settings_with_no_event_log(self):
        class Settings(PersistenceSettings):
            id_provider = UuidProvider()
            key_builder = StringDelimitedKeyBuilder()
            database = InMemoryDatabase(key_builder=key_builder)

        return Settings

    def _settings_with_event_log(self):
        class Settings(PersistenceSettings):
            id_provider = UuidProvider()
            key_builder = StringDelimitedKeyBuilder()
            database = InMemoryDatabase(key_builder=key_builder)
            event_log = EventLog(
                path=self.event_log_path,
                channel=InMemoryChannel())

        return Settings

    def _model(self, slice_size, settings):
        STFT = stft(resample_to=SR11025(), store_fft=True)

        class Model(STFT, settings):
            binary = ArrayWithUnitsFeature(
                Binarize,
                predicate=lambda data: data >= 0,
                needs=STFT.fft,
                store=False)

            sliced = ArrayWithUnitsFeature(
                Slice,
                sl=slice(0, slice_size),
                needs=binary,
                store=True)

        return Model

    def _index(self, document, feature):
        return HammingIndex(
            document,
            feature,
            path=self.hamming_db_path)
