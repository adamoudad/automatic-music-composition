import numpy as np

def generate_note(seed, model, pitchnames):
    # Preprocessing
    prediction_input = np.reshape(seed, (1, len(seed), 1)) / float(len(pitchnames))
    # Prediction
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    return pitchnames[index]


