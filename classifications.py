import numpy as np
from constants import CONFIG, SAMPLE_NUM_PER_LABEL, LABELS, LABEL_TO_NUM, ENTRY_TYPE_REVERSE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import datetime
import pytz
import pathlib
import matplotlib.pyplot as plt
from io import StringIO
import os
from collections import defaultdict


def calc_centers(x, y, model):
    output = model.predict_on_batch(x)
    centers = [[] for _ in range(CONFIG["num_classes"])]

    for i in range(len(y)):
        centers[y[i]].append(output[i])

    centers = np.array([np.mean(center, axis=0) for center in centers])

    return centers


def calc_distance(x, x_support, y_support, model):
    output = model.predict_on_batch(x)
    # centers, weights = calc_centers(x_support, y_support, model)

    # d_list = np.array(
    #     [
    #         [
    #             np.linalg.norm(
    #                 vect - center
    #             )
    #             for center in centers
    #         ] / weights
    #         for vect in output
    #     ]
    # )
    # d_list = np.array(
    #     [
    #         d / np.sum(d)
    #         for d in d_list
    #     ]
    # )

    coordinates = model.predict_on_batch(x_support)

    d_list = np.array(
        [
            [
                np.linalg.norm(
                    vect - coordinate
                )
                for coordinate in coordinates
            ]
            for vect in output
        ]
    )

    return d_list


def calc_distances(args):
    index, x, x_support, y_support, models = args

    preds = np.array(
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
            )
            for j in range(CONFIG["epochs"])
        ]
    )

    return preds


def load_distances(index):
    print("Load Distance " + str(index + 1) + "/" + str(CONFIG["num_models"]))

    distance = [
        np.load(
            "./temp/model_" + str(index) + "_epoch_" + str(epoch) + ".npy"
        )
        for epoch in range(CONFIG["epochs"])
    ]
    return distance


def load_losses(index):
    print("Load Loss " + str(index + 1) + "/" + str(CONFIG["num_models"]))

    losses = np.load(
        "./temp/model_" + str(index) + "_losses" + ".npy"
    )
    return losses


def calc_pred(x, x_support, y_support, model):
    d_list = calc_distance(x, x_support, y_support, model)

    pred = np.argmin(d_list, axis=1)

    return pred


def calc_preds(args):
    index, x, x_support, y_support, models = args

    print(
        "Calculate Prediction "
        + str(index + 1) + "/" + str(CONFIG["num_models"])
    )

    preds = np.array(
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

    return preds


def accuracy_scores(args):
    index, y, preds = args

    print(
        "Calculate Accuracy "
        + str(index + 1) + "/" + str(CONFIG["num_models"])
    )

    acc_list = np.array(
        [
            accuracy_score(
                y,
                preds[epoch],
            )
            for epoch in range(CONFIG["epochs"])
        ]
    )

    return acc_list


def calc_ensemble_accuracy(x, y, y_orig, y_support, p, e_i):
    print("-" * 200)

    # distances = np.array(p.map(load_distances, range(CONFIG["num_models"])))
    distances = np.array(
        [
            load_distances(i)
            for i in range(CONFIG["num_models"])
        ]
    )

    print("-" * 200)

    args = [
        [
            i,
            y,
            # np.argmin(distances[i], axis=2)
            np.array(
                [
                    y_support[idx]
                    for idx in np.argmin(distances[i], axis=2)
                ]
            )
        ]
        for i in range(CONFIG["num_models"])
    ]
    acc_list = np.array(p.map(accuracy_scores, args))

    print("-" * 200)

    for model_index in range(CONFIG["num_models"]):
        print(
            "Model: " + str(model_index + 1) + "/" + str(CONFIG["num_models"])
            + "\tTest Accuracy: "
            + "\tLast: " + "{:.07f}".format(acc_list[model_index][-1])
            + "\tAverage: " + "{:.07f}".format(np.mean(acc_list[model_index]))
              + " ± " + "{:.07f}".format(np.std(acc_list[model_index]))
            + "\tMin: " + "{:.07f}".format(np.min(acc_list[model_index]))
            + "\tMax: " + "{:.07f}".format(np.max(acc_list[model_index]))
            + "\tEnsemble Accuracy: " + "{:.07f}".format(
              accuracy_score(
                  y,
                  [
                      # np.argmin(
                      #     np.sum(
                      #         distances[i, :, j],
                      #         axis=0
                      #     )
                      # )
                      np.array(
                          [
                              y_support[
                                  np.argmin(
                                      np.sum(
                                        distances[model_index, :, class_num],
                                          axis=0
                                      )
                                  )
                              ]
                          ]
                      )
                      for class_num in range(distances.shape[2])
                  ]
              )
            )
        )
        pred = y_support[np.argmin(distances[model_index, -1], axis=1)]
        c_mat = confusion_matrix(y, pred)
        print(c_mat)

    print("-" * 200)

    print(
        "Accuracy Summary: " + "{:.07f}".format(np.mean(acc_list[:, -1]))
        + " ± " + "{:.07f}".format(np.std(acc_list[:, -1]))
        + "\tMin: " + "{:.07f}".format(np.min(acc_list[:, -1]))
        + "\tMax: " + "{:.07f}".format(np.max(acc_list[:, -1]))
    )

    print("-" * 200)

    result = {}

    ensemble_acc_list = np.array([])

    for epoch in range(CONFIG["epochs"]):
        pred = [
            # np.argmin(
            #     np.sum(
            #         distances[:, i, j],
            #         axis=0
            #     )
            # )
            np.array(
                [
                    y_support[
                        np.argmin(
                            np.sum(
                                distances[:, epoch, support_index],
                                axis=0
                            )
                        )
                    ]
                ]
            )
            for support_index in range(distances.shape[2])
        ]
        acc = accuracy_score(y, pred)
        ensemble_acc_list = np.append(ensemble_acc_list, acc)
        if (epoch + 1) % 10 == 0:
            print(
                "Epoch: " + str(epoch + 1) + "/" +
                str(CONFIG["epochs"])
                + "\tEnsemble Accuracy: " + "{:.07f}".format(acc)
            )
        if epoch == CONFIG["epochs"] - 1:
            report = classification_report(y, pred, target_names=LABELS)
            print(report)
            report2_correct = defaultdict(int)
            report2_total = defaultdict(int)
            for pr, yo in zip(pred, y_orig):
                if pr == LABEL_TO_NUM[ENTRY_TYPE_REVERSE[yo]]:
                    report2_correct[yo] += 1
                else:
                    report2_correct[yo] += 0
                report2_total[yo] += 1
            for label in ENTRY_TYPE_REVERSE:
                if report2_total[label] > 0:
                    print(
                        label.ljust(20, " "),
                        report2_correct[label],
                        report2_total[label],
                        "{:.04f}".format(
                            report2_correct[label] / report2_total[label]
                        ),
                        sep='\t\t'
                    )

            c_mat = confusion_matrix(y, pred)
            print(c_mat)
            # save_report(acc, report, c_mat, "Last Ensemble")  # models[0][0])
            result["last"] = classification_report(
                y,
                pred,
                output_dict=True,
                target_names=LABELS
            )

    print("-" * 200)

    cm = plt.get_cmap("jet", CONFIG["num_models"] + 1)

    plt.figure(figsize=(12, 8))
    x = list(range(1, CONFIG["epochs"] + 1))
    for model_index in range(CONFIG["num_models"]):
        plt.plot(x, acc_list[model_index], label="Model %s" %
                 (model_index + 1), c=cm(model_index))
    plt.plot(x, ensemble_acc_list, label="Ensemble",
             c=cm(CONFIG["num_models"]))
    plt.xlabel("Epoch")
    plt.xlim(0, CONFIG["epochs"])
    plt.xticks(
        np.arange(
            0,
            CONFIG["epochs"] + 1,
            # CONFIG["epochs"],
            25,
        )
    )
    plt.ylabel("Accuracy")
    plt.ylim(0.80, 1.00)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=10)
    file_name = "result_" + str(e_i) + ".jpg"
    if os.path.isfile(file_name):
        os.remove(file_name)
    plt.savefig(file_name)

    plt.clf()

    losses = np.array(p.map(load_losses, range(CONFIG["num_models"])))

    plt.figure(figsize=(12, 8))
    x = list(range(1, CONFIG["epochs"] + 1))
    for i in range(CONFIG["num_models"]):
        plt.plot(x, losses[i], label="Model %s" % (i + 1), c=cm(i))
    plt.xlabel("Epoch")
    plt.xlim(0, CONFIG["epochs"])
    plt.xticks(
        np.arange(
            0,
            CONFIG["epochs"] + 1,
            # CONFIG["epochs"],
            25,
        )
    )
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=10)
    file_name = "loss_result_" + str(e_i) + ".jpg"
    if os.path.isfile(file_name):
        os.remove(file_name)
    plt.savefig(file_name)

    plt.clf()
    plt.close()

    return result


def save_report(acc, report, c_mat, title="", model=None):
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
