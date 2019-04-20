import unittest2
from .midi import midi_to_note, note_to_midi


class MidiTests(unittest2.TestCase):

    def test_midi_to_note_0(self):
        self.assertEqual('C0', midi_to_note(0))

    def test_midi_to_note_127(self):
        self.assertEqual('G10', midi_to_note(127))

    def test_midi_to_note_33(self):
        self.assertEqual('A2', midi_to_note(33))

    def test_midi_to_note_99(self):
        self.assertEqual('D#8', midi_to_note(99))

    def test_note_to_midi_c0(self):
        self.assertEqual(0, note_to_midi('C0'))

    def test_note_to_midi_g10(self):
        self.assertEqual(127, note_to_midi('G10'))

    def test_note_to_midi_a_sharp_7(self):
        self.assertEqual(94, note_to_midi('A#7'))
