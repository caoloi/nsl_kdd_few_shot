import tensorflow as tf
import multiprocessing as mp
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras import backend as K
from losses import center_loss
from summary import create_summary, print_summary, save_summary
from models import build_fsl_attention, build_fsl_cnn, build_fsl_dnn
from callbacks import Histories
from data_processing import create_csv
from classifications import calc_ensemble_accuracy
from constants import CONFIG

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def __train(args):
    model_index, x_train, x_support, x_test, y_train, y_support, _, y_train_value, y_support_value, y_test_value = args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_index % 2)
    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=[
                0.8,  # 1
                0.4,  # 2
                0.25,  # 3
                0.4,  # 4
                0.15,  # 5
                0.24,  # 6
                0.1,  # 7
                0.1,  # 8
                0.1,  # 9
                0.1,  # 10
                0.1,  # 11
                0.09,  # 12
            ][CONFIG["num_process"] - 1],
        )
    )
    session = tf.compat.v1.Session(config=config)
    K.set_session(session)

    print(
        "Setting up Model "
        + str(model_index + 1) + "/" + str(CONFIG["num_models"])
    )

    input = Input(shape=CONFIG["input_shape"])
    output = build_fsl_attention(
        input
    ) if CONFIG["model_type"] == "cnn" else build_fsl_dnn(input)
    model = Model(inputs=input, outputs=output)

    model.compile(
        optimizer=Adam(),
        loss=center_loss(
            x_support,
            y_support_value,
            model
        ),
    )

    expanded_y_train = np.array(
        [
            np.concatenate(
                [
                    y,
                    np.full(
                        CONFIG["output_dim"] - y_train.shape[1],
                        0.0,
                    )
                ]
            ) for y in y_train
        ]
    )

    histories = Histories(
        x_train,
        y_train_value,
        x_test,
        y_test_value,
        x_support,
        y_support_value,
        model_index,
    )

    init = tf.compat.v1.global_variables_initializer()
    session.run(init)

    model.fit(
        x_train,
        expanded_y_train,
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        verbose=False,
        callbacks=[
            histories
        ],
        shuffle=CONFIG["shuffle"],
    )

    K.clear_session()


def __train_and_create_result(p, e_i):
    dir_name = CONFIG["benchmark_dir"] + "/support/" + \
        CONFIG["test_sampling_method"] + "/" + str(e_i) + "/"
    x_support = np.load(dir_name + "x_support.npy")
    y_support = np.load(dir_name + "y_support.npy")
    y_support_value = np.load(dir_name + "y_support_value.npy")

    dir_name = CONFIG["benchmark_dir"] + "/test/"
    x_test = np.load(dir_name + "x_test.npy")
    y_test = np.load(dir_name + "y_test.npy")
    y_test_value = np.load(dir_name + "y_test_value.npy")
    y_test_orig = np.load(dir_name + "y_test_orig.npy", allow_pickle=True)

    args = []

    for model_index in range(CONFIG["num_models"]):
        # dir_name = CONFIG["benchmark_dir"] + "/train/" + \
        #     ["a", "b", "c", "d", "e", "f"][model_index % 6] + \
        #     "/" + str(e_i) + "/" + str(model_index) + "/"
        # dir_name = CONFIG["benchmark_dir"] + "/train/" + "a" + "/" + \
        #     str(e_i) + "/" + str(model_index) + "/"
        dir_name = CONFIG["benchmark_dir"] + "/train/" + \
            ["a", "d", "f"][model_index % 3] + \
            "/" + str(e_i) + "/" + str(model_index) + "/"
        x_train = np.load(dir_name + "x_train.npy")
        y_train = np.load(dir_name + "y_train.npy")
        y_train_value = np.load(dir_name + "y_train_value.npy")

        # support_ids = np.random.permutation(x_support.shape[0])
        # support_ids = np.tile(
        #     support_ids,
        #     10,
        # )
        # random_x_support = x_support[support_ids]
        # random_y_support = y_support[support_ids]
        # random_y_support_value = y_support_value[support_ids]

        # train_ids = np.random.permutation(x_train.shape[0])
        # random_x_train = x_train[train_ids]
        # random_y_train = y_train[train_ids]
        # random_y_train_value = y_train_value[train_ids]

        # x_train = np.vstack((random_x_train, random_x_support))
        # y_train = np.vstack((random_y_train, random_y_support))
        # y_train_value = np.hstack(
        #     (random_y_train_value, random_y_support_value))

        # del support_ids, random_x_support, random_y_support, random_y_support_value, train_ids, random_x_train, random_y_train, random_y_train_value

        args.append(
            [
                model_index,
                x_train,
                x_support,
                x_test,
                y_train,
                y_support,
                y_test,
                y_train_value,
                y_support_value,
                y_test_value,
            ]
        )
    p.map(__train, args)

    result = calc_ensemble_accuracy(
        x_test,
        y_test_value,
        y_test_orig,
        y_support_value,
        p,
        e_i,
    )

    return result


def __main():
    p = mp.get_context('spawn').Pool(CONFIG["num_process"])
    results = []

    for i in range(CONFIG["experiment_count"]):
        # create_csv()
        print("-" * 200)
        print(
            "Experiment "
            + str(i + 1)
            + "/"
            + str(CONFIG["experiment_count"])
        )
        result = __train_and_create_result(p, i)
        results.append(result)

    summary = create_summary(results)
    print_summary(summary)
    save_summary(summary)


if __name__ == "__main__":
    __main()
    # create_csv()
