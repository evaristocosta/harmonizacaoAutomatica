import time
import tensorflow as tf

from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Nadam, Adamax


from tensorflow.keras.models import Sequential
import talos
from talos.metrics.keras_metrics import f1score

import matplotlib.pyplot as plt


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


p = {
    "shapes": ["funnel"],
    "hidden_layers": [0],
    "first_neuron": [32],
    "activation": ["selu"],
    "dropout": [0.15],
    "dense_1": [64],
    "lr": [0.01],
    "optimizer": [Adam],
    "batch_size": [16],
    "epochs": [150],
    "weight_regulizer": [None],
    "emb_output_dims": [None],
}

scan_object = talos.Scan(
    X_train,
    Y_train,
    x_val=X_val,
    y_val=Y_val,
    params=p,
    model=modelo,
    experiment_name=arq,
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


agora = str(int(time.time()))
arq = 'mlp'

# access the summary details
print("\nDetails:")
print(scan_object.details)

# use Scan object as input
analyze_object = talos.Analyze(scan_object)

# get the highest result for any metric
print("\nLowest val_loss:")
print(analyze_object.low("val_loss"))

print("\nHighest val_accuracy:")
print(analyze_object.high("val_accuracy"))

print("\nHighest val_f1score:")
print(analyze_object.high("val_f1score"))

print("\nCorrelation (val_loss):")
print(
    analyze_object.correlate(
        "val_loss",
        ["accuracy", "loss", "val_accuracy", "f1score", "val_f1score"],
    )
)

analyze_object.plot_kde("dropout", "val_loss")
plt.savefig(arq + "/" + agora + "_kde_dropout.png")

if arq == "mlp":
    """analyze_object.plot_bars("first_neuron", "val_loss", "shapes", "hidden_layers")
    plt.savefig(arq + "/" + agora + "_relacao_hidden_layers.png")"""

    analyze_object.plot_bars("lr", "val_loss", "optimizer", "batch_size")
    plt.savefig(arq + "/" + agora + "_relacao_fit.png")

    """ analyze_object.plot_box("activation", "val_loss", "first_neuron")
    plt.savefig(arq + "/" + agora + "_activation_neuronios.png") """

    """ analyze_object.plot_box("activation", "val_loss", "dense_1")
    plt.savefig(arq + "/" + agora + "_activation_neuronios_saida.png") """

    analyze_object.plot_box("epochs", "val_loss", "batch_size")
    plt.savefig(arq + "/" + agora + "_epochs_batch.png")

else:
    analyze_object.plot_bars("lr", "val_loss", "optimizer", "batch_size")
    plt.savefig(arq + "/" + agora + "_relacao_fit.png")

    analyze_object.plot_box("dense_1", "val_loss", "dense_2")
    plt.savefig(arq + "/" + agora + "_neuronios.png")

    analyze_object.plot_box("batch_size", "val_loss", "model_choice_1")
    plt.savefig(arq + "/" + agora + "_batch_size_modelos.png")

    analyze_object.plot_box("batch_size", "val_loss", "model_choice_2")
    plt.savefig(arq + "/" + agora + "_batch_size_profundidade.png")

    analyze_object.plot_box("batch_size", "val_loss", "epochs")
    plt.savefig(arq + "/" + agora + "_batch_size_epochs.png")

    analyze_object.plot_box("lr", "val_loss", "epochs")
    plt.savefig(arq + "/" + agora + "_lr_epochs.png")

    analyze_object.plot_box("dense_1", "val_loss", "activation")
    plt.savefig(arq + "/" + agora + "_activation_neuronios_camada_1.png")

    analyze_object.plot_box("dense_2", "val_loss", "activation")
    plt.savefig(arq + "/" + agora + "_activation_neuronios_camada_2.png")

    """ analyze_object.plot_box("dense_3", "val_loss", "activation")
    plt.savefig(arq + "/" + agora + "_activation_neuronios_camada_3.png") """

analyze_object.plot_corr(
    "val_loss",
    ["accuracy", "loss", "val_accuracy", "f1score", "val_f1score"],
)
plt.savefig(arq + "/" + agora + "_1_correlacao_parametros.png")

analyze_object.plot_regs("val_loss", "loss")
plt.savefig(arq + "/" + agora + "_2_regs_loss.png")

analyze_object.plot_regs("val_accuracy", "accuracy")
plt.savefig(arq + "/" + agora + "_3_regs_accuracy.png")

analyze_object.plot_kde("val_loss")
plt.savefig(arq + "/" + agora + "_4_distribuicao_loss.png")

analyze_object.plot_kde("val_accuracy")
plt.savefig(arq + "/" + agora + "_5_distribuicao_accuracy.png")

analyze_object.plot_line("val_loss")
plt.savefig(arq + "/" + agora + "_6_historico_loss.png")

analyze_object.plot_line("val_accuracy")
plt.savefig(arq + "/" + agora + "_7_historico_accuracy.png")
