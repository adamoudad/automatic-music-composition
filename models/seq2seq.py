from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, TimeDistributed, Cropping1D
from keras.layers import LSTM, Lambda


def seq2seq_lstm(sequence_length, output_length, n_classes, embedding_size=128):
    # Input layer
    model = Sequential()
    
    model.add(LSTM(embedding_size, return_sequences=True, input_shape=(sequence_length, n_classes)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, go_backwards=True,return_sequences=True))
    if output_length < sequence_length:
        model.add(Cropping1D(cropping=(sequence_length-output_length,0)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(n_classes)))
    # model.add(Lambda(lambda x: x / SOFTMAX_TEMPERATURE)) # Add temperature to softmax
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    return model

