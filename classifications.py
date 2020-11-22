from keras import backend as K
import numpy as np
from constants import CONFIG, SAMPLE_NUM_PER_LABEL
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from multiprocessing import Pool, cpu_count
from time import sleep
import datetime
import pathlib
import matplotlib.pyplot as plt


def calc_centers(x, y, model):
  output = model.predict(x)
  centers = [[] for _ in range(CONFIG["num_classes"])]

  for xx, yy in zip(output, y):
    centers[yy].append(xx)

  centers = np.array(centers)
  centers = np.array([np.mean(center, axis=0) for center in centers])

  return centers


def calc_centers_2(pred, true):
  centers = [[] for _ in range(CONFIG["num_classes"])]

  for xx, yy in zip(pred, true):
    centers[np.argmax(yy)].append(xx)

  centers = np.array(centers)

  centers = np.array([np.mean(center, axis=0) for center in centers])

  return centers


def calc_distance(x, x_support, y_support, model):
  output = model.predict(x)
  centers = calc_centers(x_support, y_support, model)

  d_list = np.array(
      [
          [
              np.linalg.norm(
                  vect - center
              )
              for center in centers
          ]
          for vect in output
      ]
  )
  d_list = np.array(
      [
          d / np.sum(d)
          for d in d_list
      ]
  )

  return d_list


def calc_distances(args):
  index, x, x_support, y_support, models = args

  sleep(index % cpu_count() * 0.01)

  preds = np.array(
      [
          (
              print(
                  "Calculate Distances" +
                  "\tEpoch: " + str(j + 1) + "/" + str(CONFIG["epochs"])
                  + "\tModel: " + str(index + 1) + "/" + str(CONFIG["num_models"])
              ) if (
                  (j + 1) % 10 == 0
                  and
                  (index + 1) == CONFIG["num_models"]
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

  return preds, index


def calc_pred(x, x_support, y_support, model):
  d_list = calc_distance(x, x_support, y_support, model)

  pred = np.argmin(d_list, axis=1)

  return pred


def calc_preds(args):
  index, x, x_support, y_support, models = args

  sleep(index % cpu_count() * 0.01)
  print("Calculate Prediction " + str(index + 1) + "/" + str(CONFIG["num_models"]))

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

  return preds, index


def accuracy_scores(args):
  index, y, preds = args

  sleep(index % cpu_count() * 0.01)
  print("Calculate Accuracy " + str(index + 1) + "/" + str(CONFIG["num_models"]))

  acc_list = np.array(
      [
          accuracy_score(
              y,
              preds[j],
          )
          for j in range(CONFIG["epochs"])
      ]
  )

  return acc_list, index


def calc_ensemble_accuracy(x, y, x_support, y_support, models):
  p = Pool(CONFIG["num_models"])

  print("-" * 200)

  args = [
      [
          i,
          x,
          x_support,
          y_support,
          models[i]
      ]
      for i in range(CONFIG["num_models"])
  ]
  distances = np.array(p.map(calc_distances, args))
  distances = np.array(sorted(distances, key=lambda x: x[-1]))
  distances = np.array([distances[i][0] for i in range(CONFIG["num_models"])])

  print("-" * 200)

  args = [
      [
          i,
          y,
          np.argmin(distances[i], axis=2)
      ]
      for i in range(CONFIG["num_models"])
  ]
  acc_list = np.array(p.map(accuracy_scores, args))
  acc_list = np.array(sorted(acc_list, key=lambda x: x[-1]))
  acc_list = np.array([acc_list[i][0] for i in range(CONFIG["num_models"])])

  print("-" * 200)

  for i in range(CONFIG["num_models"]):
    print(
        "Model: " + str(i + 1) + "/" + str(CONFIG["num_models"])
        + "\tTest Accuracy: "
        + "\tLast: " + "{:.07f}".format(acc_list[i][-1])
        + "\tAverage: " + "{:.07f}".format(np.mean(acc_list[i])) + " ± " + "{:.07f}".format(np.std(acc_list[i]))
        + "\tMin: " + "{:.07f}".format(np.min(acc_list[i]))
        + "\tMax: " + "{:.07f}".format(np.max(acc_list[i]))
        + "\tEnsemble Accuracy: " + "{:.07f}".format(
          accuracy_score(
              y,
              [
                  np.argmin(
                      np.sum(
                          distances[i, :, j],
                          axis=0
                      )
                  )
                  for j in range(distances.shape[2])
              ]
          )
        )
    )

  print("-" * 200)

  print(
      "Accuracy Summary: " + "{:.07f}".format(np.mean(acc_list[:, -1])) + " ± " + "{:.07f}".format(np.std(acc_list[:, -1]))
      + "\tMin: " + "{:.07f}".format(np.min(acc_list[:, -1]))
      + "\tMax: " + "{:.07f}".format(np.max(acc_list[:, -1]))
  )

  print("-" * 200)

  for i in range(CONFIG["epochs"]):
    pred = [
        np.argmin(
            np.sum(
                distances[:, i, j],
                axis=0
            )
        )
        for j in range(distances.shape[2])
    ]
    acc = accuracy_score(y, pred)
    print(
        "Epoch: " + str(i + 1) + "/" + str(CONFIG["epochs"])
        + "\tEnsemble Accuracy: " + "{:.07f}".format(acc)
    )
    if i == CONFIG["epochs"] - 1:
      report = classification_report(y, pred)
      print(report)
      c_mat = confusion_matrix(y, pred)
      print(c_mat)
      save_report(acc, report, c_mat, "Ensemble")

  print("-" * 200)

  distances = np.array(
      [
          distances[i][j]
          for j in range(CONFIG["epochs"])
          for i in range(CONFIG["num_models"])
      ]
  )
  pred = [
      np.argmin(
          np.sum(
              distances[:, i],
              axis=0
          )
      )
      for i in range(distances.shape[1])
  ]
  acc = accuracy_score(y, pred)
  print("Ensemble Accuracy:\t" + "{:.07f}".format(acc))
  report = classification_report(y, pred)
  print(report)
  c_mat = confusion_matrix(y, pred)
  print(c_mat)
  save_report(acc, report, c_mat, "Ensemble")

  x = list(range(1, CONFIG["epochs"] + 1))
  for i in range(CONFIG["num_models"]):
    plt.plot(x, acc_list[i], label="Model %s" % (i + 1))
  plt.xlabel("Epoch")
  plt.xlim(0, CONFIG["epochs"])
  plt.xticks(np.arange(0, CONFIG["epochs"] + 1, 5))
  plt.ylabel("Accuracy")
  plt.grid(True)
  plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=10)
  plt.show()


def save_report(acc, report, c_mat, title=""):
  now = datetime.datetime.now()
  dir = "./results/" + "{:.07f}".format(acc)[2:4] + "/" + now.strftime("%Y%m%d")
  if not pathlib.Path(dir).exists():
    pathlib.Path(dir).mkdir(parents=True)
  file = pathlib.Path(dir + "/" + "{:.07f}".format(acc)[2:6] + "_" + now.strftime("%Y%m%d_%H%M%S.txt"))
  with file.open(mode="w") as f:
    print(title, file=f)
    print(now.strftime("%Y%m%d_%H%M%S"), file=f)
    print("Accuracy: " + str(acc), file=f)
    print("Classification Report:", file=f)
    print(report, file=f)
    print("Confusion Matrix:", file=f)
    print(c_mat, file=f)
    print(CONFIG, file=f)
    print(SAMPLE_NUM_PER_LABEL, file=f)
