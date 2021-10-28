import shutil
import os
import numpy as np
from data_processing import create_csv, train_data_processing, support_data_processing, test_data_processing
from constants import CONFIG


def __create_benchmark_support_dataset():
    for test_sampling_method in ["e_025", "e_050", "e_075", "e_100"]:
        CONFIG["test_sampling_method"] = test_sampling_method
        for benchmark_index in range(10):
            x_support, y_support, y_support_value = support_data_processing(
                [
                    benchmark_index,
                    "zero",
                ]
            )
            dir_name = CONFIG["benchmark_dir"] + "/support/" + \
                CONFIG["test_sampling_method"] + \
                "/" + str(benchmark_index) + "/"
            if os.path.isdir(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)
            np.savetxt(
                dir_name + "x_support.csv",
                x_support,
                delimiter=',',
                fmt='%.8f',
            )
            np.savetxt(
                dir_name + "y_support.csv",
                y_support,
                delimiter=',',
                fmt='%d',
            )
            np.savetxt(
                dir_name + "y_support_value.csv",
                y_support_value,
                delimiter=',',
                fmt='%d',
            )


def __create_benchmark_train_dataset():
    methods = ["a", "b", "c", "d", "e", "f"]
    for benchmark_index in range(10):
        for method_index in range(6):
            method = methods[method_index]
            for model_index in range(CONFIG["num_models"]):
                x_train, y_train, y_train_value = train_data_processing(
                    [
                        benchmark_index * method_index * model_index,
                        method
                    ]
                )
                dir_name = CONFIG["benchmark_dir"] + "/train/" + method + \
                    "/" + str(benchmark_index) + "/" + str(model_index) + "/"
                if os.path.isdir(dir_name):
                    shutil.rmtree(dir_name)
                os.makedirs(dir_name)
                np.savetxt(
                    dir_name + "x_train.csv",
                    x_train,
                    delimiter=',',
                    fmt='%.8f',
                )
                np.savetxt(
                    dir_name + "y_train.csv",
                    y_train,
                    delimiter=',',
                    fmt='%d',
                )
                np.savetxt(
                    dir_name + "y_train_value.csv",
                    y_train_value,
                    delimiter=',',
                    fmt='%d',
                )


def __create_benchmark_test_dataset():
    x_test, y_test, y_test_value, y_test_orig = test_data_processing(
        [
            None,
            "zero",
        ]
    )
    dir_name = CONFIG["benchmark_dir"] + "/test/"
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    np.savetxt(
        dir_name + "x_test.csv",
        x_test,
        delimiter=',',
        fmt='%.8f',
    )
    np.savetxt(
        dir_name + "y_test.csv",
        y_test,
        delimiter=',',
        fmt='%d',
    )
    np.savetxt(
        dir_name + "y_test_value.csv",
        y_test_value,
        delimiter=',',
        fmt='%d',
    )
    np.savetxt(
        dir_name + "y_test_orig.csv",
        y_test_orig,
        delimiter=',',
        fmt='%s',
    )


if __name__ == "__main__":
    create_csv()
    __create_benchmark_support_dataset()
    __create_benchmark_train_dataset()
    __create_benchmark_test_dataset()
