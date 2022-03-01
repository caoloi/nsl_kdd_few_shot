import tensorflow.keras as keras
from sklearn.metrics import classification_report, accuracy_score
from classifications import calc_distance
from constants import CONFIG, LABELS
import matplotlib.pyplot as plt
import numpy as np
from save_result_utils import save_d_list, save_loss, save_center_distances, save_average_distances


class Histories(keras.callbacks.Callback):
    def __init__(self, x_train, y_train, x_test, y_test, x_support, y_support, benchmark_index, model_index):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_support = x_support
        self.y_support = y_support
        self.benchmark_index = benchmark_index
        self.model_index = model_index

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        d_list, centers = calc_distance(
            self.x_test,
            self.x_support,
            self.y_support,
            self.model,
        )
        center_distances = []
        for label_i in range(CONFIG["num_classes"]):
            for label_j in range(CONFIG["num_classes"]):
                if label_i < label_j:
                    center_i = centers[label_i]
                    center_j = centers[label_j]
                    center_distance = np.linalg.norm(
                        center_i - center_j
                    )
                    center_distances.append(
                        "{:.04f}".format(center_distance)
                    )

        average_distances = []
        for label_i in range(CONFIG["num_classes"]):
            d_list_per_class = d_list[np.where(self.y_support == label_i)]
            average_distances.append(np.mean(d_list_per_class[:, label_i]))

        if (epoch + 1) % np.min([25, CONFIG["epochs"]]) == 0:
            pred = np.argmin(d_list, axis=1)
            # pred = np.array(
            #     [
            #         self.y_support[idx]
            #         for idx in np.argmin(d_list, axis=1)
            #     ]
            # )
            acc = accuracy_score(self.y_test, pred)
            # print(acc)
            print(
                "Epoch: " + str(epoch + 1) + "/" + str(CONFIG["epochs"])
                + "\tModel: " + str(self.model_index + 1) +
                "/" + str(CONFIG["num_models"])
                + "\tTest Accuracy: " + "|  " * self.model_index
                + "{:.07f}".format(acc) + "|  "
                * (CONFIG["num_models"] - self.model_index - 1)
                + "\tLoss: " + "|  " * self.model_index
                + "{:.04f}".format(logs["loss"]).zfill(8) + "|  "
                * (CONFIG["num_models"] - self.model_index - 1)
                + " " + " ".join(center_distances)
            )
            if False:
                plt.ion()
                plt.clf()
                cmp = plt.get_cmap("jet", CONFIG["num_classes"])
                # plt.figure(figsize=(8, 8))
                for i in range(CONFIG["num_classes"]):
                    output = self.model.predict(self.x_test)
                    plt.scatter(output[self.y_test == i, 0],
                                output[self.y_test == i, 1], color=cmp(i), marker=f"${i}$")
                plt.title(
                    "Model: " + str(self.model_index + 1) + "/" +
                    str(CONFIG["num_models"]) + ", "
                    + "Epoch: " + str(epoch + 1) + "/" +
                    str(CONFIG["epochs"]) + ", "
                    + "Accuracy: " + "{:.02f}".format(acc * 100)
                )
                plt.draw()
                plt.pause(0.001)
            if acc >= 0.955:  # or epoch == CONFIG["epochs"] - 1:
                report = classification_report(
                    self.y_test, pred, target_names=LABELS)
                # c_mat = confusion_matrix(self.y_test, pred)
                print(report)
                # save_report(acc, report, c_mat, "Epoch: " + str(epoch), self.model)

        save_d_list(
            self.benchmark_index,
            self.model_index,
            epoch,
            d_list
        )

        save_loss(
            self.benchmark_index,
            self.model_index,
            epoch,
            logs["loss"]
        )

        save_center_distances(
            self.benchmark_index,
            self.model_index,
            epoch,
            center_distances
        )

        save_average_distances(
            self.benchmark_index,
            self.model_index,
            epoch,
            average_distances
        )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
