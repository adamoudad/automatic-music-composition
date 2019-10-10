from os import path
import numpy as np
import glob

from music21 import converter, instrument
from music21.note import Note
from music21.chord import Chord

from settings import NOTE_DICTIONARY, DURATION_DICTIONARY, MAX_LENGTH, STEP, N_SCALES, N_DURATIONS, N_CLASSES, MIDI_PATH

def load_midi_to_array(midi_path=MIDI_PATH):
    """
    <https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5>
    
    :param data_path: Path directory containing all midi files
    :type data_path: str
    
    :return: list of notes
    :rtype: list
    """
    notes = []

    for file in glob.glob(path.join(midi_path,"*.mid")):
        midi = converter.parse(file)
        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, Note):
                notes.append(str(element.pitch))
            elif isinstance(element, Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def get_network_input_output(notes):
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(len(pitchnames))  # replaced n_vocab by len(pitchnames)

    return network_input, network_output, pitchnames

