import numpy as np
from constants import CONFIG, SAMPLE_NUM_PER_LABEL
from sklearn.metrics import accuracy_score
import datetime
import pytz
import pathlib
from io import StringIO
from tensorflow.compat.v1.keras import backend as K
import tensorflow as tf
from save_result_utils import average_distances_file_name, d_list_file_name, loss_file_name, center_distances_file_name
from generate_result_figure_utils import generate_accuracy_result_jpg, generate_loss_result_jpg, generate_center_distances_result_jpg, generate_average_distances_result_jpg
from display_summary_utils import display_accuracy_summary, display_ensemble_summary, display_model_summary


def calc_centers(x, y, model):
    output = model.predict_on_batch(x)
    centers = [[] for _ in range(CONFIG["num_classes"])]

    for i in range(len(y)):
        centers[y[i]].append(output[i])

    centers = np.array([np.mean(center, axis=0) for center in centers])

    return centers


def calc_centers_for_support_tensor(support_true, support_pred):
    support_true_value = K.argmax(support_true)
    centers = []

    for i in range(CONFIG["num_classes"]):
        indices = tf.where(tf.equal(support_true_value, i))
        pred_per_class = tf.gather_nd(support_pred, indices=indices)
        mean_pred = K.mean(pred_per_class, axis=0)
        centers.append(mean_pred)

    return centers


def calc_distance(x, x_support, y_support, model):
    output = model.predict_on_batch(x)
    centers = calc_centers(x_support, y_support, model)

    d_list = np.array(
        [
            [
                np.linalg.norm(
                    vector - center
                )
                for center in centers
            ]
            for vector in output
        ]
    )
    d_list = np.array(
        [
            d / np.sum(d)
            for d in d_list
        ]
    )

    # coordinates = model.predict_on_batch(x_support)

    # d_list = np.array(
    #     [
    #         [
    #             np.linalg.norm(
    #                 vector - coordinate
    #             )
    #             for coordinate in coordinates
    #         ]
    #         for vector in output
    #     ]
    # )

    return d_list, centers


def calc_distances(args):
    index, x, x_support, y_support, models = args

    predictions = np.array(
        [
            (
                print(
                    "Calculate Distances"
                    + "\tEpoch: " + str(j + 1) + "/" + str(CONFIG["epochs"])
                    + "\tModel: " + str(index + 1) + "/" +
                    str(CONFIG["num_models"])
                ) if (
                    (j + 1) % 25 == 0
                    and
                    (index + 1) % CONFIG["num_process"] == 0
                ) else False
            )
            or
            calc_distance(
                x,
                x_support,
                y_support,
                models[j]
            )[0]
            for j in range(CONFIG["epochs"])
        ]
    )

    return predictions


def load_distances(args):
    benchmark_index, model_index = args

    print("Load Distance " + str(model_index + 1) +
          "/" + str(CONFIG["num_models"]))

    distance = [
        np.loadtxt(
            d_list_file_name(benchmark_index, model_index, epoch),
            delimiter=',',
            dtype='float'
        )
        for epoch in range(CONFIG["epochs"])
    ]
    return distance


def load_losses(args):
    benchmark_index, model_index = args

    print("Load Loss " + str(model_index + 1) +
          "/" + str(CONFIG["num_models"]))

    losses = np.loadtxt(
        loss_file_name(benchmark_index, model_index),
        delimiter=',',
        dtype='float'
    )
    return losses


def load_center_distances(args):
    benchmark_index, model_index = args

    print("Load Center Distances " + str(model_index + 1) +
          "/" + str(CONFIG["num_models"]))

    center_distances = np.loadtxt(
        center_distances_file_name(benchmark_index, model_index),
        delimiter=',',
        dtype='float'
    )

    return center_distances


def load_average_distances(args):
    benchmark_index, model_index = args

    print("Load Average Distances " + str(model_index + 1) +
          "/" + str(CONFIG["num_models"]))

    average_distances = np.loadtxt(
        average_distances_file_name(benchmark_index, model_index),
        delimiter=',',
        dtype='float'
    )

    return average_distances


def calc_pred(x, x_support, y_support, model):
    d_list, _ = calc_distance(x, x_support, y_support, model)

    pred = np.argmin(d_list, axis=1)

    return pred


def calc_predictions(args):
    index, x, x_support, y_support, models = args

    print(
        "Calculate Prediction "
        + str(index + 1) + "/" + str(CONFIG["num_models"])
    )

    predictions = np.array(
        [
            calc_pred(
                x,
                x_support,
                y_support,
                models[j]
            )
            for j in range(CONFIG["epochs"])
        ]
    )

    return predictions


def accuracy_scores(args):
    index, y, predictions = args

    print(
        "Calculate Accuracy "
        + str(index + 1) + "/" + str(CONFIG["num_models"])
    )

    acc_list = np.array(
        [
            accuracy_score(
                y,
                predictions[epoch],
            )
            for epoch in range(CONFIG["epochs"])
        ]
    )

    return acc_list


def calc_ensemble_accuracy(x, y, y_orig, y_support, p, benchmark_index):
    print("-" * 200)

    # distances = np.array(p.map(load_distances, range(CONFIG["num_models"])))
    # distances = np.array(
    #     [
    #         load_distances(benchmark_index, model_index)
    #         for model_index in range(CONFIG["num_models"])
    #     ]
    # )

    distances = np.array(
        p.map(
            load_distances,
            [
                [benchmark_index, model_index]
                for model_index in range(CONFIG["num_models"])
            ]
        )
    )

    print("-" * 200)

    acc_list = np.array(
        p.map(
            accuracy_scores,
            [
                [
                    i,
                    y,
                    np.argmin(distances[i], axis=2)
                    # np.array(
                    #     [
                    #         y_support[idx]
                    #         for idx in np.argmin(distances[i], axis=2)
                    #     ]
                    # )
                ]
                for i in range(CONFIG["num_models"])
            ]
        )
    )

    print("-" * 200)

    display_model_summary(y, y_support, acc_list, distances)

    print("-" * 200)

    display_accuracy_summary(acc_list)

    print("-" * 200)

    result = {}

    result, ensemble_acc_list = display_ensemble_summary(
        y, y_orig, distances, result
    )

    print("-" * 200)

    generate_accuracy_result_jpg(
        benchmark_index, ensemble_acc_list, acc_list)

    losses = np.array(
        p.map(
            load_losses,
            [
                [benchmark_index, model_index]
                for model_index in range(CONFIG["num_models"])
            ]
        )
    )

    generate_loss_result_jpg(benchmark_index, losses)

    center_distances = np.array(
        p.map(
            load_center_distances,
            [
                [benchmark_index, model_index]
                for model_index in range(CONFIG["num_models"])
            ]
        )
    )

    generate_center_distances_result_jpg(benchmark_index, center_distances)

    average_distances = np.array(
        p.map(
            load_average_distances,
            [
                [benchmark_index, model_index]
                for model_index in range(CONFIG["num_models"])
            ]
        )
    )

    generate_average_distances_result_jpg(benchmark_index, average_distances)

    return result


def save_report(acc, report, c_mat, title="", model=None) -> None:
    if CONFIG["save_report"]:
        now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        dir = "./results/" + \
            "{:.07f}".format(acc)[2:4] + "/" + now.strftime("%Y%m%d")
        if not pathlib.Path(dir).exists():
            pathlib.Path(dir).mkdir(parents=True)
        file = pathlib.Path(
            dir + "/" + "{:.07f}".format(acc)[2:8] +
            "_" + now.strftime("%Y%m%d_%H%M%S.txt")
        )
        with file.open(mode="w") as f:
            print(now.strftime("%Y%m%d_%H%M%S"), file=f)
            print(title, file=f)
            print("CONFIG:", file=f)
            print(CONFIG, file=f)
            print("SAMPLE_NUM_PER_LABEL:", file=f)
            print(SAMPLE_NUM_PER_LABEL, file=f)
            print("Accuracy: " + str(acc), file=f)
            print("Classification Report:", file=f)
            print(report, file=f)
            print("Confusion Matrix:", file=f)
            print(c_mat, file=f)
            if model is not None:
                print("Model Summary:", file=f)
                with StringIO() as buf:
                    model.summary(print_fn=lambda x: buf.write(x + "\n"))
                    text = buf.getvalue()

                print(text, file=f)

    return
