from tensorflow.keras.applications import VGG16, ResNet101, InceptionV3, DenseNet201
from tensorflow.keras.applications import vgg16, resnet, inception_v3, densenet
from keras.models import Model
from keras.layers import Dense


def preprocess(X_train, X_val, X_test, params):
    X_train *= 255
    X_val *= 255
    X_test *= 255

    if params["application"] == "vgg16":
        X_train = vgg16.preprocess_input(X_train)
        X_val = vgg16.preprocess_input(X_val)
        X_test = vgg16.preprocess_input(X_test)
    elif params["application"] == "resnet101":
        X_train = resnet.preprocess_input(X_train)
        X_val = resnet.preprocess_input(X_val)
        X_test = resnet.preprocess_input(X_test)
    elif params["application"] == "inceptionv3":
        X_train = inception_v3.preprocess_input(X_train)
        X_val = inception_v3.preprocess_input(X_val)
        X_test = inception_v3.preprocess_input(X_test)
    elif params["application"] == "densenet201":
        X_train = densenet.preprocess_input(X_train)
        X_val = densenet.preprocess_input(X_val)
        X_test = densenet.preprocess_input(X_test)

    return X_train, X_val, X_test


def model(params):
    if params["application"] == "vgg16":
        application = VGG16(
            include_top=False,
            weights=None,
            input_shape=params["input_shape"],
            pooling="max",
        )
    elif params["application"] == "resnet101":
        application = ResNet101(
            include_top=False,
            weights=None,
            input_shape=params["input_shape"],
            pooling="max",
        )
    elif params["application"] == "inceptionv3":
        application = InceptionV3(
            include_top=False,
            weights=None,
            input_shape=params["input_shape"],
            pooling="max",
        )
    elif params["application"] == "densenet201":
        application = DenseNet201(
            include_top=False,
            weights=None,
            input_shape=params["input_shape"],
            pooling="max",
        )

    output = Dense(params["output_shape"], activation="softmax")(application.output)
    model = Model(application.input, output)

    return model, model.get_weights()
