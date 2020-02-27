import unittest2
from .hammingdb import HammingDb
from uuid import uuid4
import shutil
import numpy as np
import os


class HammingDbTests(unittest2.TestCase):
    def setUp(self):
        self._db_name = uuid4().hex
        self._path = '/tmp/{path}'.format(path=self._db_name)

        def extract_code_from_text(text, n_chunks=1):
            n_bits = n_chunks * 64
            code = np.zeros(n_bits, dtype=np.bool)
            for i in range(len(text)):
                trigram = text[i: i + 3]
                hashed = hash(trigram)
                code[hashed % n_bits] = 1
            return np.packbits(code).tostring()

        self.extract_code_from_text = extract_code_from_text

    def tearDown(self):
        try:
            shutil.rmtree(self._path)
        except OSError:
            pass

    def test_can_initialize_with_no_code_size_when_data_already_exists(self):
        HammingDb(self._path, code_size=32)
        db2 = HammingDb(self._path, code_size=None)
        self.assertEqual(32, db2.code_size)

    def test_raises_when_persisted_code_size_does_not_agree_with_init(self):
        HammingDb(self._path, code_size=32)
        self.assertRaises(
            ValueError, lambda: HammingDb(self._path, code_size=16))

    def test_raises_when_code_size_is_none_for_uninitialized_database(self):
        self.assertRaises(
            ValueError, lambda: HammingDb(self._path, code_size=None))

    def test_metadata_fetch_does_not_raise_when_unitialized(self):
        db = HammingDb(self._path, code_size=16)
        v = db.get_metadata(b'cat')
        self.assertEqual(None, v)

    def test_can_set_and_get_metadata(self):
        db = HammingDb(self._path, code_size=16)
        db.set_metadata(b'cat', b'dog')
        self.assertEqual(b'dog', db.get_metadata(b'cat'))

    def test_can_get_random_entry(self):
        db = HammingDb(self._path, code_size=16)
        for i in range(100):
            db.append(os.urandom(16), str(i))
        code, results = db.random_search(10)
        results = list(results)
        self.assertEqual(10, len(results))

    def test_can_create_database_with_128_bit_codes(self):
        db = HammingDb(self._path, code_size=16)
        self.assertEqual(0, len(db))

    def test_can_append_with_128_bits(self):
        db = HammingDb(self._path, code_size=16)
        db.append('a' * 16, 'some data')
        self.assertEqual(1, len(db))

    def test_can_search_with_128_bits(self):
        db = HammingDb(self._path, code_size=16)
        t1 = b'Mary had a little lamb'
        t2 = b'Mary had a little dog'
        t3 = b'Permanent Midnight'
        t4 = b'Mary sad a little cog'
        extract_code = lambda x: self.extract_code_from_text(x, n_chunks=2)
        db.append(extract_code(t1), t1)
        db.append(extract_code(t2), t2)
        db.append(extract_code(t3), t3)
        db.append(extract_code(t4), t4)
        results = list(db.search(extract_code(t1), 3))
        self.assertEqual(3, len(results))
        self.assertEqual(t1, results[0])
        self.assertEqual(t2, results[1])
        self.assertEqual(t4, results[2])

    def test_can_run_in_write_only_mode(self):
        db = HammingDb(self._path, code_size=16, writeonly=True)
        t1 = 'Mary had a little lamb'
        t2 = 'Mary had a little dog'
        t3 = 'Permanent Midnight'
        t4 = 'Mary sad a little cog'
        extract_code = lambda x: self.extract_code_from_text(x, n_chunks=2)
        db.append(extract_code(t1), t1)
        db.append(extract_code(t2), t2)
        db.append(extract_code(t3), t3)
        db.append(extract_code(t4), t4)
        self.assertIsNone(db._codes)

    def test_search_raises_in_write_only_mode(self):
        db = HammingDb(self._path, code_size=16, writeonly=True)
        t1 = 'Mary had a little lamb'
        t2 = 'Mary had a little dog'
        t3 = 'Permanent Midnight'
        t4 = 'Mary sad a little cog'
        extract_code = lambda x: self.extract_code_from_text(x, n_chunks=2)
        db.append(extract_code(t1), t1)
        db.append(extract_code(t2), t2)
        db.append(extract_code(t3), t3)
        db.append(extract_code(t4), t4)
        self.assertRaises(
            RuntimeError, lambda: list(db.search(extract_code(t1), 3)))

    def test_constructor_raises_when_not_multiple_of_eight(self):
        self.assertRaises(
            ValueError, lambda: HammingDb(self._path, code_size=7))

    def test_empty_db_has_zero_length(self):
        db = HammingDb(self._path, code_size=8)
        self.assertEqual(0, len(db))

    def test_db_has_length_one_after_appending(self):
        db = HammingDb(self._path, code_size=8)
        db.append('a' * 8, 'some data')
        self.assertEqual(1, len(db))

    def test_db_has_length_two_after_appending_twice(self):
        db = HammingDb(self._path, code_size=8)
        db.append('a' * 8, 'some data')
        db.append('a' * 8, 'some data')
        self.assertEqual(2, len(db))

    @unittest2.skip('This test fails in CI tests, but it is soon to be removed')
    def test_can_search_over_text_documents(self):
        db = HammingDb(self._path, code_size=8)
        t1 = b'Mary had a little lamb'
        t2 = b'Mary had a little dog'
        t3 = b'Permanent Midnight'
        t4 = b'Mary sad a little cog'
        db.append(self.extract_code_from_text(t1), t1)
        db.append(self.extract_code_from_text(t2), t2)
        db.append(self.extract_code_from_text(t3), t3)
        db.append(self.extract_code_from_text(t4), t4)
        results = list(db.search(self.extract_code_from_text(t1), 3))
        self.assertEqual(3, len(results))
        self.assertEqual(t1, results[0])
        self.assertEqual(t2, results[1])
        self.assertEqual(t4, results[2])

    def test_can_search_over_data_added_from_another_instance(self):
        db = HammingDb(self._path, code_size=8)
        db2 = HammingDb(self._path, code_size=8)
        t1 = b'Mary had a little lamb'
        t2 = b'Mary had a little dog'
        t3 = b'Permanent Midnight'
        t4 = b'Mary sad a little cog'
        db.append(self.extract_code_from_text(t1), t1)
        db.append(self.extract_code_from_text(t2), t2)
        db.append(self.extract_code_from_text(t3), t3)
        db.append(self.extract_code_from_text(t4), t4)
        results = list(db2.search(self.extract_code_from_text(t1), 3))
        self.assertEqual(3, len(results))
        s = set(results)
        self.assertTrue(t1 in s)
        self.assertTrue(t2 in s)
        self.assertTrue(t4 in s)

    def test_cannot_append_wrong_code_size(self):
        db = HammingDb(self._path, code_size=8)
        self.assertRaises(ValueError, lambda: db.append('a' * 7, 'some data'))

    def test_cannot_search_for_wrong_code_size(self):
        db = HammingDb(self._path, code_size=8)
        self.assertRaises(ValueError, lambda: list(db.search('a' * 7, 10)))

    def test_external_modifications_are_detected(self):
        db = HammingDb(self._path, code_size=8)
        db2 = HammingDb(self._path, code_size=8)
        db2.append('a' * 8, 'some data')
        self.assertEqual(1, len(db))

    def test_external_modifications_are_detected_when_db_has_keys(self):
        db = HammingDb(self._path, code_size=8)
        db2 = HammingDb(self._path, code_size=8)
        db.append('a' * 8, 'some other data')
        db2.append('a' * 8, 'some data')
        self.assertEqual(2, len(db))
        self.assertEqual(2, len(db2))

    def test_db_starts_with_correct_number_of_keys(self):
        db2 = HammingDb(self._path, code_size=8)
        db2.append('a' * 8, 'some data')
        db = HammingDb(self._path, code_size=8)
        self.assertEqual(1, len(db))

    def test_can_retrieve_data_from_search(self):
        db = HammingDb(self._path, code_size=8)
        t1 = b'Mary had a little lamb'
        t2 = b'Mary had a little dog'
        t3 = b'Permanent Midnight'
        t4 = b'Mary sad a little cog'
        db.append(self.extract_code_from_text(t1), t1)
        db.append(self.extract_code_from_text(t2), t2)
        db.append(self.extract_code_from_text(t3), t3)
        db.append(self.extract_code_from_text(t4), t4)
        results = list(db.search(self.extract_code_from_text(t1), 3))
        data = results[0]
        self.assertEqual(t1, data)
