import time
import joblib
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Nadam, Adamax
import talos

import matplotlib.pyplot as plt

from load_data import carrega, separa, carrega_arquivo
from model_runner import model_fit


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

    input_shape = "{0}".format(str(X_train.shape[1:]))
    output_shape = Y_train.shape[1]

    p = {
        # "shapes": ["funnel"],
        # "hidden_layers": [0],
        # "first_neuron": [32],
        "model": ["alexnet_optimization"],
        "input_shape": [input_shape],
        "output_shape": [output_shape],
        "neurons": [256, 128, 64],
        "activation": ["relu", "elu"],
        "dropout": (0.25, 0.55, 6),
        "layer_4": [True, False],
        "dense_1": [3072, 2048, 1024],
        "dense_2": [4096, 2048, 1024],
        "learning_rate": [0.1 * 100], # [0.001, 0.01, 0.0001]
        "optimizer": [SGD], # [Adam, SGD, RMSprop]
        "momentum": [0.9], # [0.9]
        "batch_size": [128], # [64, 128]
        "epochs": [50], # (10, 100, 9)
        "weight_regulizer": [None],
        "emb_output_dims": [None],
    }

    scan_object = talos.Scan(
        X_train,
        Y_train,
        x_val=X_val,
        y_val=Y_val,
        params=p,
        model=model_fit,
        experiment_name="alexnet_optimization",
        fraction_limit=0.1,
        # round_limit=30,
        # reduction_method="kendall",
        # reduction_interval=4,
        # reduction_window=2,
        # reduction_threshold=0.2,
        # reduction_metric="val_loss",
        # minimize_loss=True,
        # random_method="uniform_mersenne",
        # seed=42,
        clear_session=True,
        save_weights=False,
    )

    # access the summary details
    print("\nDetails:")
    print(scan_object.details)

    print("\nData:")
    print(scan_object.data)

    # use Scan object as input
    analyze_object = talos.Analyze(scan_object)

    joblib.dump(analyze_object, "analyze_object.pkl")

    print_optimization_details(analyze_object)
    plot_optimization_results(analyze_object)

    return analyze_object


def print_optimization_details(analyze_object):
    # get the highest result for any metric
    print("\nLowest val_loss:")
    print(analyze_object.low("val_loss"))

    print("\nHighest val_accuracy:")
    print(analyze_object.high("val_accuracy"))

    print("\nCorrelation (val_loss):")
    print(
        analyze_object.correlate(
            "val_loss",
            ["accuracy", "loss", "val_accuracy"],
        )
    )


def plot_optimization_results(analyze_object):
    arq = "alexnet_optimization"
    analyze_object.plot_kde("dropout", "val_loss")
    plt.savefig(arq + "/kde_dropout.png")
    # plt.show()

    analyze_object.plot_bars("learning_rate", "val_loss", "optimizer", "batch_size")
    plt.savefig(arq + "/relacao_fit.png")
    # plt.show()

    analyze_object.plot_box("dense_1", "val_loss", "dense_2")
    plt.savefig(arq + "/neuronios.png")
    # plt.show()

    analyze_object.plot_box("batch_size", "val_loss", "layer_4")
    plt.savefig(arq + "/batch_size_layer_adicional.png")
    # plt.show()

    analyze_object.plot_box("batch_size", "val_loss", "epochs")
    plt.savefig(arq + "/batch_size_epochs.png")
    # plt.show()

    analyze_object.plot_box("learning_rate", "val_loss", "epochs")
    plt.savefig(arq + "/lr_epochs.png")
    # plt.show()

    analyze_object.plot_box("dense_1", "val_loss", "activation")
    plt.savefig(arq + "/activation_neuronios_camada_1.png")
    # plt.show()

    analyze_object.plot_box("dense_2", "val_loss", "activation")
    plt.savefig(arq + "/activation_neuronios_camada_2.png")
    # plt.show()

    """ analyze_object.plot_box("dense_3", "val_loss", "activation")
    plt.savefig(arq + "/activation_neuronios_camada_3.png") """
    # plt.show()

    analyze_object.plot_regs("val_loss", "loss")
    plt.savefig(arq + "/2_regs_loss.png")
    # plt.show()

    analyze_object.plot_regs("val_accuracy", "accuracy")
    plt.savefig(arq + "/3_regs_accuracy.png")
    # plt.show()

    analyze_object.plot_kde("val_loss")
    plt.savefig(arq + "/4_distribuicao_loss.png")
    # plt.show()

    analyze_object.plot_kde("val_accuracy")
    plt.savefig(arq + "/5_distribuicao_accuracy.png")
    # plt.show()

    analyze_object.plot_line("val_loss")
    plt.savefig(arq + "/6_historico_loss.png")
    # plt.show()

    analyze_object.plot_line("val_accuracy")
    plt.savefig(arq + "/7_historico_accuracy.png")
    # plt.show()

    analyze_object.plot_corr(
        "val_loss",
        ["accuracy", "loss", "val_accuracy"],
    )
    plt.savefig(arq + "/1_correlacao_parametros.png")
    # plt.show()


if __name__ == "__main__":
    optimizer()
