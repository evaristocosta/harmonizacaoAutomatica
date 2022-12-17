from keras.models import Sequential
from keras.layers import Input, Dense, Activation, LSTM, Bidirectional


def model(params):
    model = Sequential()
    model.add(Input(shape=(1, params["input_shape"])))
    model.add(
        Bidirectional(LSTM(params["neurons"], return_sequences=True, activation=None))
    )
    model.add(Activation(params["activation"]))
    model.add(Bidirectional(LSTM(params["neurons"], activation=None)))
    model.add(Activation(params["activation"]))
    model.add(Dense(params["output_shape"]))
    model.add(Activation("softmax"))

    return model, model.get_weights()
