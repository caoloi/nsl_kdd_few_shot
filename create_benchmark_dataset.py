import shutil
import os
import numpy as np
from data_processing import train_data_processing, support_data_processing, test_data_processing
from constants import CONFIG


def __create_benchmark_support_dataset():
    for e_i in range(10):
        x_support, y_support, y_support_value = support_data_processing(
            [
                None,
                "zero",
            ]
        )
        dir_name = "./benchmark/support/" + \
            CONFIG["test_sampling_method"] + "/" + str(e_i) + "/"
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        np.save(
            dir_name + "x_support",
            x_support,
        )
        np.save(
            dir_name + "y_support",
            y_support,
        )
        np.save(
            dir_name + "y_support_value",
            y_support_value,
        )


def __create_benchmark_train_dataset():
    for e_i in range(10):
        for method in ["a", "b", "c", "d", "e", "f"]:
            for i in range(CONFIG["num_models"]):
                x_train, y_train, y_train_value = train_data_processing(
                    [
                        i,
                        method
                    ]
                )
                dir_name = "./benchmark/train/" + method + \
                    "/" + str(e_i) + "/" + str(i) + "/"
                if os.path.isdir(dir_name):
                    shutil.rmtree(dir_name)
                os.makedirs(dir_name)
                np.save(
                    dir_name + "x_train",
                    x_train,
                )
                np.save(
                    dir_name + "y_train",
                    y_train,
                )
                np.save(
                    dir_name + "y_train_value",
                    y_train_value,
                )


def __create_benchmark_test_dataset():
    x_test, y_test, y_test_value, y_test_orig = test_data_processing(
        [
            None,
            "zero",
        ]
    )
    dir_name = "./benchmark/test/"
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    np.save(
        dir_name + "x_test",
        x_test,
    )
    np.save(
        dir_name + "y_test",
        y_test,
    )
    np.save(
        dir_name + "y_test_value",
        y_test_value,
    )
    np.save(
        dir_name + "y_test_orig",
        y_test_orig,
    )


if __name__ == "__main__":
    __create_benchmark_support_dataset()
    __create_benchmark_train_dataset()
    __create_benchmark_test_dataset()
