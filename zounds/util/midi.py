NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_INDICES = {n: i for i, n in enumerate(NOTES)}


def midi_to_note(midi_number):
    octave = midi_number // len(NOTES)
    note_name = NOTES[int(midi_number % len(NOTES))]
    return f'{note_name}{octave}'


def note_to_midi(note):
    if note[1] == '#':
        note_name_length = 2
    else:
        note_name_length = 1

    note_name, octave = note[:note_name_length], int(note[note_name_length:])
    return (octave * len(NOTES)) + NOTE_INDICES[note_name]
