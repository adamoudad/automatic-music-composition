from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Lambda


def two_layers_lstm(sequence_length, input_size, n_classes, dropout=0.3):
    '''
    Baseline model. Two layers LSTM.
    <https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5>
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x/n_classes, input_shape=(sequence_length, input_size)))  # Normalize input
    model.add(LSTM(
        256,
        return_sequences=True
    ))
    model.add(Dropout(dropout))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(dropout))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
