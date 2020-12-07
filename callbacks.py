import keras
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from classifications import calc_pred, save_report
from constants import CONFIG
import matplotlib.pyplot as plt


class Histories(keras.callbacks.Callback):
  def __init__(self, x_train, y_train, x_test, y_test, x_support, y_support, index):
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.x_support = x_support
    self.y_support = y_support
    self.index = index

  def on_train_begin(self, logs={}):
    self.aucs = []
    self.losses = []

  def on_train_end(self, logs={}):
    return

  def on_epoch_begin(self, epoch, logs={}):
    return

  def on_epoch_end(self, epoch, logs={}):
    # print("Model: " + str(self.index + 1) + "/" + str(CONFIG["num_models"]))
    # print("========= train =========")
    # train_pred = calc_pred(self.x_train, self.x_support, self.y_support, self.model)
    # print(classification_report(self.y_train, train_pred))
    # train_acc = accuracy_score(self.y_train, train_pred)
    # print(train_acc)

    # print("========= test =========")
    pred = calc_pred(self.x_test, self.x_support, self.y_support, self.model)
    acc = accuracy_score(self.y_test, pred)
    # print(acc)
    print(
        "Epoch: " + str(epoch + 1) + "/" + str(CONFIG["epochs"])
        + "\tModel: " + str(self.index + 1) + "/" + str(CONFIG["num_models"])
        # + "\tTrain Accuracy: " + "|  " * self.index + "{:.07f}".format(train_acc) + "|  " * (CONFIG["num_models"] - self.index - 1)
        + "\tTest Accuracy: " + "|  " * self.index + "{:.07f}".format(acc) + "|  " * (CONFIG["num_models"] - self.index - 1)
        + "\t\tLoss: " + "|  " * self.index + "{:.07f}".format(logs["loss"]) + "|  " * (CONFIG["num_models"] - self.index - 1)
    )
    # if self.index == 0:
    if False:
      plt.ion()
      plt.clf()
      cmp = plt.get_cmap("jet", CONFIG["num_classes"])
      # plt.figure(figsize=(8, 8))
      for i in range(CONFIG["num_classes"]):
        output = self.model.predict(self.x_test)
        plt.scatter(output[self.y_test == i, 0], output[self.y_test == i, 1], color=cmp(i), marker=f"${i}$")
      plt.title(
          "Model: " + str(self.index + 1) + "/" + str(CONFIG["num_models"]) + ", "
          + "Epoch: " + str(epoch + 1) + "/" + str(CONFIG["epochs"]) + ", "
          + "Accuracy: " + "{:.02f}".format(acc * 100)
      )
      plt.draw()
      plt.pause(0.001)
    if acc >= 0.95: # or epoch == CONFIG["epochs"] - 1:
      report = classification_report(self.y_test, pred)
      c_mat = confusion_matrix(self.y_test, pred)
      save_report(acc, report, c_mat, "Epoch: " + str(epoch), self.model)

    # print("=========")

    self.model.save(
        "./temp/model_" + str(self.index) + "_epoch_" + str(epoch) + ".h5",
        include_optimizer=False,
    )

    return

  def on_batch_begin(self, batch, logs={}):
    return

  def on_batch_end(self, batch, logs={}):
    return
