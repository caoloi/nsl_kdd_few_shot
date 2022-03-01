import numpy as np
import matplotlib.pyplot as plt
import os
from constants import CONFIG, LABELS


def generate_accuracy_result_jpg(benchmark_index: int, ensemble_acc_list, acc_list) -> None:
    color_map = plt.get_cmap("jet", CONFIG["num_models"] + 1)

    plt.figure(figsize=(12, 8))
    x = list(range(1, CONFIG["epochs"] + 1))
    for model_index in range(CONFIG["num_models"]):
        plt.plot(
            x,
            acc_list[model_index],
            label="Model %s" % (model_index + 1),
            c=color_map(model_index)
        )
    plt.plot(
        x,
        ensemble_acc_list,
        label="Ensemble",
        c=color_map(CONFIG["num_models"])
    )
    plt.xlabel("Epoch")
    plt.xlim(0, CONFIG["epochs"])
    plt.xticks(
        np.arange(
            0,
            CONFIG["epochs"] + 1,
            10,
        )
    )
    plt.ylabel("Accuracy")
    plt.ylim(0.70, 1.00)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=10)

    dir_name = "result_fig/" + str(CONFIG["experiment_id"]) + "/accuracy/"
    file_name = dir_name + "benchmark_" + str(benchmark_index) + ".jpg"

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if os.path.isfile(file_name):
        os.remove(file_name)

    plt.savefig(file_name)

    plt.clf()

    plt.close()

    return


def generate_loss_result_jpg(benchmark_index: int, losses) -> None:
    color_map = plt.get_cmap("jet", CONFIG["num_models"] + 1)

    plt.figure(figsize=(12, 8))
    x = list(range(1, CONFIG["epochs"] + 1))
    for model_index in range(CONFIG["num_models"]):
        plt.plot(
            x,
            losses[model_index],
            label="Model %s" % (model_index + 1),
            c=color_map(model_index)
        )
    plt.xlabel("Epoch")
    plt.xlim(0, CONFIG["epochs"])
    plt.xticks(
        np.arange(
            0,
            CONFIG["epochs"] + 1,
            10,
        )
    )
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(
        bbox_to_anchor=(1, 1),
        loc="upper left",
        fontsize=10
    )

    dir_name = "result_fig/" + str(CONFIG["experiment_id"]) + "/loss/"
    file_name = dir_name + "benchmark_" + str(benchmark_index) + ".jpg"

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    if os.path.isfile(file_name):
        os.remove(file_name)

    plt.savefig(file_name)

    plt.clf()

    plt.close()

    return


def generate_center_distances_result_jpg(benchmark_index: int, center_distances) -> None:
    color_map = plt.get_cmap("jet", CONFIG["num_models"] + 1)

    center_distance_index = 0

    for label_i in range(CONFIG["num_classes"]):
        for label_j in range(CONFIG["num_classes"]):
            if label_i < label_j:
                plt.figure(figsize=(12, 8))
                x = list(range(1, CONFIG["epochs"] + 1))
                for model_index in range(CONFIG["num_models"]):
                    plt.plot(
                        x,
                        center_distances[
                            model_index,
                            :,
                            center_distance_index
                        ],
                        label="Model %s" % (model_index + 1),
                        c=color_map(model_index)
                    )
                plt.xlabel("Epoch")
                plt.xlim(0, CONFIG["epochs"])
                plt.xticks(
                    np.arange(
                        0,
                        CONFIG["epochs"] + 1,
                        10,
                    )
                )
                plt.ylabel(
                    "Center Distance between " +
                    LABELS[label_i] + " and " + LABELS[label_j]
                )
                plt.grid(True)
                plt.legend(
                    bbox_to_anchor=(1, 1),
                    loc="upper left",
                    fontsize=10
                )

                dir_name = "result_fig/" + \
                    str(CONFIG["experiment_id"]) + "/center_distance/" + \
                    "/benchmark_" + str(benchmark_index) + "/"
                file_name = dir_name + "between_" + \
                    LABELS[label_i] + "_and_" + LABELS[label_j] + ".jpg"

                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)
                if os.path.isfile(file_name):
                    os.remove(file_name)

                plt.savefig(file_name)

                plt.clf()
                plt.close()

                center_distance_index += 1

    return


def generate_average_distances_result_jpg(benchmark_index: int, average_distances) -> None:
    color_map = plt.get_cmap("jet", CONFIG["num_models"] + 1)

    for label_i in range(CONFIG["num_classes"]):
        plt.figure(figsize=(12, 8))
        x = list(range(1, CONFIG["epochs"] + 1))
        for model_index in range(CONFIG["num_models"]):
            plt.plot(
                x,
                average_distances[
                    model_index,
                    :,
                    label_i
                ],
                label="Model %s" % (model_index + 1),
                c=color_map(model_index)
            )
        plt.xlabel("Epoch")
        plt.xlim(0, CONFIG["epochs"])
        plt.xticks(
            np.arange(
                0,
                CONFIG["epochs"] + 1,
                10,
            )
        )
        plt.ylabel(
            "Average Distance of " + LABELS[label_i]
        )
        plt.grid(True)
        plt.legend(
            bbox_to_anchor=(1, 1),
            loc="upper left",
            fontsize=10
        )

        dir_name = "result_fig/" + \
            str(CONFIG["experiment_id"]) + "/average_distance/" + \
            "/benchmark_" + str(benchmark_index) + "/"
        file_name = dir_name + LABELS[label_i] + ".jpg"

        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        if os.path.isfile(file_name):
            os.remove(file_name)

        plt.savefig(file_name)

        plt.clf()
        plt.close()

    return
