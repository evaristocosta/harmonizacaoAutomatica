import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Nadam, Adamax
import talos

import matplotlib.pyplot as plt

from load_data import carrega
from model_runner import model_fit



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

""" gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e) """


def optimizer():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = carrega()
    input_shape = X_train.shape[1]
    output_shape = Y_train.shape[1]

    p = {
        "shapes": ["funnel"],
        "hidden_layers": [0],
        "first_neuron": [32],

        "input_shape": [input_shape],
        "output_shape": [output_shape],
        "neurons": [64, 128],
        "model": ["mlp_1_hidden"],

        "activation": ["relu", "elu"],
        "dropout": [0.15],
        "dense_1": [64],
        "learning_rate": [0.001],
        "optimizer": [Adam],
        "batch_size": [128, 256],
        "epochs": [15],
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
        experiment_name="arq",
        fraction_limit=1.0,
        # round_limit=30,
        reduction_method="pearson",
        reduction_interval=30,
        reduction_window=15,
        reduction_threshold=0.2,
        reduction_metric="val_loss",
        minimize_loss=True,
        random_method="uniform_mersenne",
        clear_session=True,
        save_weights=False,
    )

    # access the summary details
    print("\nDetails:")
    print(scan_object.details)

    # use Scan object as input
    analyze_object = talos.Analyze(scan_object)

    print_optimization_details(analyze_object)

    return analyze_object


def print_optimization_details(analyze_object):
    # get the highest result for any metric
    print("\nLowest val_loss:")
    print(analyze_object.low("val_loss"))

    print("\nHighest val_accuracy:")
    print(analyze_object.high("val_accuracy"))

    """ print("\nHighest val_f1score:")
    print(analyze_object.high("val_f1score")) """

    print("\nCorrelation (val_loss):")
    print(
        analyze_object.correlate(
            "val_loss",
            ["accuracy", "loss", "val_accuracy", "f1score", "val_f1score"],
        )
    )




def plot_optimization_results(analyze_object):
    analyze_object.plot_kde("dropout", "val_loss")
    # plt.savefig(arq + "/" + agora + "_kde_dropout.png")
    plt.show()

    arq = "mlp"
    if arq == "mlp":
        """analyze_object.plot_bars("first_neuron", "val_loss", "shapes", "hidden_layers")
    # plt.savefig(arq + "/" + agora + "_relacao_hidden_layers.png")"""
        plt.show()

        analyze_object.plot_bars("lr", "val_loss", "optimizer", "batch_size")
    # plt.savefig(arq + "/" + agora + "_relacao_fit.png")
        plt.show()

        """ analyze_object.plot_box("activation", "val_loss", "first_neuron")
    # plt.savefig(arq + "/" + agora + "_activation_neuronios.png") """
        plt.show()

        """ analyze_object.plot_box("activation", "val_loss", "dense_1")
    # plt.savefig(arq + "/" + agora + "_activation_neuronios_saida.png") """
        plt.show()

        analyze_object.plot_box("epochs", "val_loss", "batch_size")
    # plt.savefig(arq + "/" + agora + "_epochs_batch.png")
        plt.show()

    else:
        analyze_object.plot_bars("lr", "val_loss", "optimizer", "batch_size")
    # plt.savefig(arq + "/" + agora + "_relacao_fit.png")
        plt.show()

        analyze_object.plot_box("dense_1", "val_loss", "dense_2")
    # plt.savefig(arq + "/" + agora + "_neuronios.png")
        plt.show()

        analyze_object.plot_box("batch_size", "val_loss", "model_choice_1")
    # plt.savefig(arq + "/" + agora + "_batch_size_modelos.png")
        plt.show()

        analyze_object.plot_box("batch_size", "val_loss", "model_choice_2")
    # plt.savefig(arq + "/" + agora + "_batch_size_profundidade.png")
        plt.show()

        analyze_object.plot_box("batch_size", "val_loss", "epochs")
    # plt.savefig(arq + "/" + agora + "_batch_size_epochs.png")
        plt.show()

        analyze_object.plot_box("lr", "val_loss", "epochs")
    # plt.savefig(arq + "/" + agora + "_lr_epochs.png")
        plt.show()

        analyze_object.plot_box("dense_1", "val_loss", "activation")
    # plt.savefig(arq + "/" + agora + "_activation_neuronios_camada_1.png")
        plt.show()

        analyze_object.plot_box("dense_2", "val_loss", "activation")
    # plt.savefig(arq + "/" + agora + "_activation_neuronios_camada_2.png")
        plt.show()

        """ analyze_object.plot_box("dense_3", "val_loss", "activation")
    # plt.savefig(arq + "/" + agora + "_activation_neuronios_camada_3.png") """
        plt.show()

    analyze_object.plot_corr(
    "val_loss",
    ["accuracy", "loss", "val_accuracy", "f1score", "val_f1score"],
)
# plt.savefig(arq + "/" + agora + "_1_correlacao_parametros.png")
    plt.show()

    analyze_object.plot_regs("val_loss", "loss")
# plt.savefig(arq + "/" + agora + "_2_regs_loss.png")
    plt.show()

    analyze_object.plot_regs("val_accuracy", "accuracy")
# plt.savefig(arq + "/" + agora + "_3_regs_accuracy.png")
    plt.show()

    analyze_object.plot_kde("val_loss")
# plt.savefig(arq + "/" + agora + "_4_distribuicao_loss.png")
    plt.show()

    analyze_object.plot_kde("val_accuracy")
# plt.savefig(arq + "/" + agora + "_5_distribuicao_accuracy.png")
    plt.show()

    analyze_object.plot_line("val_loss")
# plt.savefig(arq + "/" + agora + "_6_historico_loss.png")
    plt.show()

    analyze_object.plot_line("val_accuracy")
# plt.savefig(arq + "/" + agora + "_7_historico_accuracy.png")
    plt.show()




if __name__ == "__main__":
    optimizer()