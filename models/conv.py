from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten


def note_convnet(sequence_length, n_classes):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(sequence_length,1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
