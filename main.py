import tensorflow as tf
import multiprocessing as mp
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras import backend as K
from tensorflow.python.client import device_lib
from losses import center_loss
from summary import create_summary, print_summary, save_summary
from models import build_fsl_attention
from callbacks import Histories
from classifications import calc_ensemble_accuracy
from constants import CONFIG
import random

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def __setup_tf_config(model_index):
    np.random.seed(model_index)
    random.seed(model_index)
    os.environ['PYTHONHASHSEED'] = str(model_index)
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
        ),
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    session = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(),
        config=config
    )
    tf.compat.v1.set_random_seed(model_index)
    K.set_session(session)

    # init = tf.compat.v1.global_variables_initializer()
    # session.run(init)


def __train(args):
    benchmark_index, model_index, x_train, x_support, x_test, y_train, y_support, _, y_train_value, y_support_value, y_test_value = args

    __setup_tf_config(model_index)

    print(
        "Setting up Model "
        + str(model_index + 1) + "/" + str(CONFIG["num_models"])
    )

    input = Input(shape=CONFIG["input_shape"])
    output = build_fsl_attention(input)
    model = Model(inputs=input, outputs=output)

    model.compile(
        optimizer=Adam(
            # lr=0.0005  # デフォルトは0.001
        ),
        loss=center_loss,
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
        benchmark_index,
        model_index,
    )

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


def __train_and_create_result(p, benchmark_index):
    dir_name = CONFIG["benchmark_dir"] + "/support/" + \
        CONFIG["test_sampling_method"] + "/" + str(benchmark_index) + "/"
    x_support = np.loadtxt(dir_name + "x_support.csv",
                           delimiter=',', dtype='float')
    y_support = np.loadtxt(dir_name + "y_support.csv",
                           delimiter=',', dtype='int')
    y_support_value = np.loadtxt(
        dir_name + "y_support_value.csv", delimiter=',', dtype='int')

    dir_name = CONFIG["benchmark_dir"] + "/test/"
    x_test = np.loadtxt(dir_name + "x_test.csv", delimiter=',', dtype='float')
    y_test = np.loadtxt(dir_name + "y_test.csv", delimiter=',', dtype='int')
    y_test_value = np.loadtxt(
        dir_name + "y_test_value.csv", delimiter=',', dtype='int')
    y_test_orig = np.loadtxt(
        dir_name + "y_test_orig.csv", delimiter=',', dtype='str')

    args = []

    for model_index in range(CONFIG["num_models"]):
        # dir_name = CONFIG["benchmark_dir"] + "/train/" + \
        #     ["a", "b", "c", "d", "e", "f"][model_index % 6] + \
        #     "/" + str(benchmark_index) + "/" + str(model_index) + "/"
        # dir_name = CONFIG["benchmark_dir"] + "/train/" + "a" + "/" + \
        #     str(benchmark_index) + "/" + str(model_index) + "/"
        dir_name = CONFIG["benchmark_dir"] + "/train/" + \
            ["a", "d", "f"][model_index % 3] + \
            "/" + str(benchmark_index) + "/" + str(model_index) + "/"
        raw_x_train = np.loadtxt(
            dir_name + "x_train.csv", delimiter=',', dtype='float')
        raw_y_train = np.loadtxt(
            dir_name + "y_train.csv", delimiter=',', dtype='int')
        raw_y_train_value = np.loadtxt(
            dir_name + "y_train_value.csv", delimiter=',', dtype='int')

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

        x_train = []
        for r_x_t in raw_x_train:
            x_train.extend([r_x_t])
            x_train.extend(x_support)
        x_train = np.array(x_train)

        y_train = []
        for r_y_t in raw_y_train:
            y_train.extend([r_y_t])
            y_train.extend(y_support)
        y_train = np.array(y_train)

        y_train_value = []
        for r_y_t_v in raw_y_train_value:
            y_train_value.extend([r_y_t_v])
            y_train_value.extend(y_support_value)
        y_train_value = np.array(y_train_value)

        args.append(
            [
                benchmark_index,
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
        benchmark_index,
    )

    return result


def __main():
    p = mp.get_context('spawn').Pool(CONFIG["num_process"])
    results = []

    for benchmark_index in range(CONFIG["experiment_count"]):
        # create_csv()
        print("-" * 200)
        print(
            "Experiment "
            + str(benchmark_index + 1)
            + "/"
            + str(CONFIG["experiment_count"])
        )
        result = __train_and_create_result(p, benchmark_index)
        results.append(result)

    summary = create_summary(results)
    print_summary(summary)
    save_summary(summary)


if __name__ == "__main__":
    __main()
    # create_csv()
