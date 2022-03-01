import os
import numpy as np
from constants import CONFIG


def save_d_list(benchmark_index: int, model_index: int, epoch: int, d_list) -> None:
    dir_name = d_list_dir_name(benchmark_index)
    file_name = d_list_file_name(
        benchmark_index,
        model_index,
        epoch
    )

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if os.path.isfile(file_name):
        os.remove(file_name)

    np.savetxt(
        file_name,
        d_list,
        delimiter=',',
        fmt='%f',
    )

    return


def save_loss(benchmark_index: int, model_index: int, epoch: int, loss) -> None:
    dir_name = loss_dir_name(benchmark_index)
    file_name = loss_file_name(
        benchmark_index,
        model_index
    )

    if epoch == 0:
        losses = np.array([loss])
    else:
        losses = np.loadtxt(
            file_name,
            delimiter=',',
            dtype='float'
        )
        losses = np.append(losses, loss)

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if os.path.isfile(file_name):
        os.remove(file_name)

    np.savetxt(
        file_name,
        losses,
        delimiter=',',
        fmt='%f',
    )

    return


def save_center_distances(benchmark_index: int, model_index: int, epoch: int, center_distances) -> None:
    dir_name = center_distances_dir_name(benchmark_index)
    file_name = dir_name + "model_" + \
        str(model_index) + "_center_distances_list.csv"
    file_name = center_distances_file_name(
        benchmark_index,
        model_index
    )

    if epoch == 0:
        center_distances_list = np.array([center_distances])
    else:
        center_distances_list = np.loadtxt(
            file_name,
            delimiter=',',
            dtype='str'
        )
        if epoch == 1:
            center_distances_list = center_distances_list.reshape(1, 10)
        center_distances_list = np.append(
            center_distances_list,
            [center_distances],
            axis=0,
        )

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if os.path.isfile(file_name):
        os.remove(file_name)

    np.savetxt(
        file_name,
        center_distances_list,
        delimiter=',',
        fmt='%s',
    )

    return


def save_average_distances(benchmark_index: int, model_index: int, epoch: int, average_distances) -> None:
    dir_name = average_distances_dir_name(benchmark_index)
    file_name = dir_name + "model_" + \
        str(model_index) + "_average_distances_list.csv"
    file_name = average_distances_file_name(
        benchmark_index,
        model_index
    )

    if epoch == 0:
        average_distances_list = np.array([average_distances])
    else:
        average_distances_list = np.loadtxt(
            file_name,
            delimiter=',',
            dtype='str'
        )
        if epoch == 1:
            average_distances_list = average_distances_list.reshape(1, 5)
        average_distances_list = np.append(
            average_distances_list,
            [average_distances],
            axis=0,
        )

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if os.path.isfile(file_name):
        os.remove(file_name)

    np.savetxt(
        file_name,
        average_distances_list,
        delimiter=',',
        fmt='%s',
    )

    return


def d_list_dir_name(benchmark_index: int) -> str:
    return "./temp/" + str(CONFIG["experiment_id"]) + "/d_list/" + str(benchmark_index) + '/'


def d_list_file_name(benchmark_index: int, model_index: int, epoch: int) -> str:
    dir_name = d_list_dir_name(benchmark_index)
    return dir_name + "model_" + str(model_index) + "_epoch_" + str(epoch) + ".csv"


def loss_dir_name(benchmark_index: int) -> str:
    return "./temp/" + str(CONFIG["experiment_id"]) + "/loss/" + str(benchmark_index) + '/'


def loss_file_name(benchmark_index: int, model_index: int) -> str:
    dir_name = loss_dir_name(benchmark_index)
    return dir_name + "model_" + str(model_index) + ".csv"


def center_distances_dir_name(benchmark_index: int) -> str:
    return "./temp/" + str(CONFIG["experiment_id"]) + "/center_distances/" + str(benchmark_index) + '/'


def center_distances_file_name(benchmark_index: int, model_index: int) -> str:
    dir_name = center_distances_dir_name(benchmark_index)
    return dir_name + "model_" + str(model_index) + ".csv"


def average_distances_dir_name(benchmark_index: int) -> str:
    return "./temp/" + str(CONFIG["experiment_id"]) + "/average_distances/" + str(benchmark_index) + '/'


def average_distances_file_name(benchmark_index: int, model_index: int) -> str:
    dir_name = average_distances_dir_name(benchmark_index)
    return dir_name + "model_" + str(model_index) + ".csv"
