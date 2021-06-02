import keras
from sklearn.metrics import classification_report, accuracy_score
from classifications import calc_distance
from constants import CONFIG, LABELS
import matplotlib.pyplot as plt
import numpy as np
import os


class Histories(keras.callbacks.Callback):
  def __init__(self, x_train, y_train, x_test, y_test, x_support, y_support, index, j):
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.x_support = x_support
    self.y_support = y_support
    self.index = index
    self.j = j

  def on_train_begin(self, logs={}):
    self.aucs = []
    self.losses = []

  def on_train_end(self, logs={}):
    return

  def on_epoch_begin(self, epoch, logs={}):
    return

  def on_epoch_end(self, epoch, logs={}):
    d_list = calc_distance(
        self.x_test,
        self.x_support,
        self.y_support,
        self.model,
    )
    if (epoch + 1) % np.min([50, CONFIG["epochs"]]) == 0:
      pred = np.argmin(d_list, axis=1)
      acc = accuracy_score(self.y_test, pred)
      # print(acc)
      print(
          "Epoch: " + str(epoch + 1 + self.j * CONFIG["epochs"])
          + "/" + str(CONFIG["epochs"] * CONFIG["repeat"])
          + "\tModel: " + str(self.index + 1) + "/" + str(CONFIG["num_models"])
          + "\tTest Accuracy: " + "|  " * self.index
          + "{:.07f}".format(acc) + "|  "
          * (CONFIG["num_models"] - self.index - 1)
          + "\t\tLoss: " + "|  " * self.index
          + "{:.07f}".format(logs["loss"]) + "|  "
          * (CONFIG["num_models"] - self.index - 1)
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
            "Model: " + str(self.index + 1) + "/" +
            str(CONFIG["num_models"]) + ", "
            + "Epoch: " + str(epoch + 1) + "/" + str(CONFIG["epochs"]) + ", "
            + "Accuracy: " + "{:.02f}".format(acc * 100)
        )
        plt.draw()
        plt.pause(0.001)
      if acc >= 0.955:  # or epoch == CONFIG["epochs"] - 1:
        report = classification_report(self.y_test, pred, target_names=LABELS)
        # c_mat = confusion_matrix(self.y_test, pred)
        print(report)
        # save_report(acc, report, c_mat, "Epoch: " + str(epoch), self.model)

    file_name = "./temp/model_" + str(
        self.index
    ) + "_epoch_" + str(
        epoch + self.j * CONFIG["epochs"]
    )
    if os.path.isfile(file_name):
      os.remove(file_name)
    np.save(
        file_name,
        d_list,
    )
    if epoch == 0 and self.j <= 0:
      losses = np.array([logs["loss"]])
    else:
      losses = np.load(
          "./temp/model_" + str(self.index) + "_losses" + ".npy"
      )
      losses = np.append(losses, logs["loss"])
    file_name = "./temp/model_" + str(self.index) + "_losses"
    if os.path.isfile(file_name):
      os.remove(file_name)
    np.save(
        file_name,
        losses,
    )

    return

  def on_batch_begin(self, batch, logs={}):
    return

  def on_batch_end(self, batch, logs={}):
    return
