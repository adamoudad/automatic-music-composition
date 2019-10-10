def split_note_sequence(notes, input_length, output_length):
    '''
    Split note sequence into a list of (input_sequence, output_sequence) pairs
    '''
    sequence = []
    for i in range(0, len(notes) - input_length - output_length):
        sequence.append((notes[i:i + input_length],
                         notes[i + input_length:
                               i + input_length + output_length])
    return sequence

