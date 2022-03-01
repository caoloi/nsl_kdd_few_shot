import numpy as np
from constants import CONFIG, LABELS, LABEL_TO_NUM, ENTRY_TYPE_REVERSE
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import defaultdict


def display_model_summary(y, y_support, acc_list, distances) -> None:
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
                      np.argmin(
                        np.sum(
                            distances[model_index, :, class_num],
                            axis=0
                        )
                      )
                      #   np.array(
                      #     [
                      #         y_support[
                      #             np.argmin(
                      #                 np.sum(
                      #                     distances[model_index, :, class_num],
                      #                     axis=0
                      #                 )
                      #             )
                      #         ]
                      #     ]
                      #   )
                      for class_num in range(distances.shape[2])
                  ]
              )
            )
        )
        pred = np.argmin(distances[model_index, -1], axis=1)
        c_mat = confusion_matrix(y, pred)
        print(c_mat)

    return


def display_accuracy_summary(acc_list) -> None:
    print(
        "Accuracy Summary: " + "{:.07f}".format(np.mean(acc_list[:, -1]))
        + " ± " + "{:.07f}".format(np.std(acc_list[:, -1]))
        + "\tMin: " + "{:.07f}".format(np.min(acc_list[:, -1]))
        + "\tMax: " + "{:.07f}".format(np.max(acc_list[:, -1]))
    )

    return


def display_ensemble_summary(y, y_orig, distances, result):
    ensemble_acc_list = np.array([])

    for epoch in range(CONFIG["epochs"]):
        pred = [
            np.argmin(
                np.sum(
                    distances[:, epoch, support_index],
                    axis=0
                )
            )
            # np.array(
            #     [
            #         y_support[
            #             np.argmin(
            #                 np.sum(
            #                     distances[:, epoch, support_index],
            #                     axis=0
            #                 )
            #             )
            #         ]
            #     ]
            # )
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
    return result, ensemble_acc_list
