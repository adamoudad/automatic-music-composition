from pretty_midi import PrettyMIDI

class MIDIScore(PrettyMIDI):
    '''
    Extension class of PrettyMIDI, fit for slicing into measures and exporting to pianoroll representation (as numpy.ndarray)
    '''
    def __init__(self, path, timestep_duration=1/4):
        super().__init__(path)
        self.timestep_duration = timestep_duration
        # A quarter note duration (in seconds) is equal to resolution (PPQN) times the duration (in s) of the first tick
        self.quarter_note_duration = self.resolution * self.tick_to_time(1) # resolution(ticks per quarter note) * tick duration(seconds per tick)
        self.sampling_frequency = self.get_sampling_frequency()

    def get_sampling_frequency(self):
        """
        Computes sampling frequency for given duration of timestep as multiplicator of a quarter note duration
        Example: timestep_duration = 1/4 gives timesteps of 16th notes
        """
        return 1 / (self.quarter_note_duration * self.timestep_duration)

    def to_pianoroll(self):
        return self.get_piano_roll(fs=self.sampling_frequency).transpose()


class Pianoroll:
    '''
    Datum representing a pianoroll; A matrix of pitch activation column-vector with fixed timestep relatively defined to the beat duration (timestep is a fraction of a beat)
    '''
    def __init__(self, matrix, timestep_duration):
        self.timestep_duration = timestep_duration
        self.matrix = np.array(matrix)

    def __add__(self, other):
        # if self.timestep_duration != other.timestep_duration:
        warnings.warn("Timestep duration mismatch, be careful, kid... Using first pianoroll's timestep duration.", Warning)
        return Pianoroll(np.concatenate((self.matrix, other.matrix)), timestep_duration=self.timestep_duration)

    def __len__(self):
        return len(self.matrix)


