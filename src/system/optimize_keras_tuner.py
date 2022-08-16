import time
import joblib
import keras_tuner
from tensorflow import keras
import tensorflow as tf


from load_data import carrega, separa, carrega_arquivo
from models import alexnet_optimization


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def optimizer():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = carrega_arquivo()
    del X_test, Y_test
    
    model = alexnet_optimization.model

    """ tuner = keras_tuner.Hyperband(
        max_epochs=10,
        hypermodel=model,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        objective="val_loss",
        directory="src/system/results",
        project_name="alexnet_optimization_keras",
    ) """

    tuner = keras_tuner.BayesianOptimization(
        hypermodel=model,
        objective="val_loss",
        distribution_strategy=tf.distribute.MirroredStrategy(),
        directory="src/system/results",
        project_name="alexnet_optimization_keras",
    )

    stop_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir="src/system/results/alexnet_optimization_keras/logs",
    )

    tuner.search_space_summary()

    tuner.search(
        X_train,
        Y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, Y_val),
        callbacks=[stop_early, tensorboard],
    )
    tuner.results_summary()

    best_model = tuner.get_best_models()[0]
    best_model.build(input_shape=(96, 96, 3))
    best_model.summary()

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps)

    model = tuner.hypermodel.build(best_hps)
    joblib.dump(model, "src/system/results/alexnet_optimization_keras/best_model.pkl")


if __name__ == "__main__":
    optimizer()
