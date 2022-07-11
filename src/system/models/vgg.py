from tensorflow.keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense


def model(params):
    vgg = VGG16(
        include_top=False,
        weights=None,
        input_shape=params["input_shape"],
        pooling="max",
    )

    output = Dense(params["output_shape"], activation="softmax")(vgg.output)
    model = Model(vgg.input, output)

    return model, model.get_weights()
