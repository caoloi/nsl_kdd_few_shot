import platform
pf = platform.system()
if pf == 'Darwin':
  import plaidml.keras
  import os
  plaidml.keras.install_backend()
  os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
  os.environ["PLAIDML_EXPERIMENTAL"] = "1"
else:
  import os
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from constants import CONFIG, SAMPLE_NUM_PER_LABEL
from classifications import calc_ensemble_accuracy
from data_processing import data_processing
from callbacks import Histories
from models import build_fsl_cnn, build_fsl_dnn
from losses import center_loss
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
from multiprocessing import Pool
import datetime
import pytz
import pathlib
import sys


def train(args):
  import platform
  pf = platform.system()
  if pf != 'Darwin':
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto(
        # allow_soft_placement=True
    )
    config.gpu_options.per_process_gpu_memory_fraction = [
        0.8,
        0.3,
        0.15,
        0.1,
    ][CONFIG["num_process"] - 1]
    # config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
  K.set_epsilon(CONFIG["epsilon"])
  K.set_floatx(CONFIG["floatx"])

  index, j, x_train, x_support, x_test, y_train, y_support, _, y_train_value, y_support_value, y_test_value, input_shape = args

  print(
      "Setting up Model "
      + str(index + 1) + "/" + str(CONFIG["num_models"])
      + "(" + str(j + 1) + ")"
  )

  input = Input(shape=input_shape)
  output = build_fsl_cnn(
      input
  ) if CONFIG["model_type"] == "cnn" else build_fsl_dnn(input)
  model = Model(inputs=input, outputs=output)

  if j > 0:
    model.load_weights("./temp/model_" + str(index) + "_" + str(j - 1) + ".h5")
  model.compile(
      optimizer=Adam(),
      loss=[
          center_loss(
              x_support,
              y_support,
              y_support_value,
              model
          )
      ],
  )

  expanded_y_train = np.array(
      [
          np.concatenate(
              [
                  y,
                  np.full(
                      CONFIG["output_dim"] - y_train.shape[1],
                      0.0,
                  )
              ]
          ) for y in y_train
      ]
  )

  histories = Histories(
      x_train,
      y_train_value,
      x_test,
      y_test_value,
      x_support,
      y_support_value,
      index,
      j,
  )

  model.fit(
      x_train,
      expanded_y_train,
      batch_size=CONFIG["batch_size"] + index * 10,
      epochs=CONFIG["epochs"],
      verbose=False,
      callbacks=[
          histories
      ],
      shuffle=CONFIG["shuffle"],
  )

  model.save_weights("./temp/model_" + str(index) + "_" + str(j) + ".h5")

  K.clear_session()


def create_summary(results):
  summary = {}

  for result in results:
    for type in result:
      for label in result[type]:
        if label == "accuracy":
          if label not in summary[type]:
            summary[type][label] = []
          summary[type][label].append(result[type][label])
        else:
          for metric in result[type][label]:
            if metric != "support":
              if type not in summary:
                summary[type] = {}
              if label not in summary[type]:
                summary[type][label] = {}
              if metric not in summary[type][label]:
                summary[type][label][metric] = []
              summary[type][label][metric].append(result[type][label][metric])

  return summary


def print_summary(summary, f=sys.stdout):
  for type in summary:
    print(type, file=f)
    for label in summary[type]:
      print("\t" + label, end="", file=f)
      if label == "accuracy":
        mean = np.mean(summary[type][label])
        std = np.std(summary[type][label])
        min = np.min(summary[type][label])
        max = np.max(summary[type][label])
        print(
            "\t\t"
            + "{:.04f}".format(mean) + " ± " + "{:.04f}".format(std)
            + " min: " + "{:.04f}".format(min)
            + " max: " + "{:.04f}".format(max),
            end="",
            file=f,
        )
      else:
        for metric in summary[type][label]:
          mean = np.mean(summary[type][label][metric])
          std = np.std(summary[type][label][metric])
          min = np.min(summary[type][label][metric])
          max = np.max(summary[type][label][metric])
          print(
              "\t\t" + metric + ": "
              + "{:.04f}".format(mean) + " ± " + "{:.04f}".format(std)
              + " min: " + "{:.04f}".format(min)
              + " max: " + "{:.04f}".format(max),
              end="",
              file=f,
          )
      print("", file=f)
    print("", file=f)


def save_summary(summary):
  if CONFIG["save_report"]:
    now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    acc = np.mean(summary["last_10"]["accuracy"])
    dir = "./summaries/" + \
        "{:.04f}".format(acc)[2:4] + "/" + now.strftime("%Y%m%d")
    if not pathlib.Path(dir).exists():
      pathlib.Path(dir).mkdir(parents=True)
    file = pathlib.Path(
        dir + "/" + "{:.04f}".format(acc)[2:6] +
        "_" + now.strftime("%Y%m%d_%H%M%S.txt")
    )
    with file.open(mode="w") as f:
      print(now.strftime("%Y%m%d_%H%M%S"), file=f)
      print("Summary", file=f)
      print("CONFIG:", file=f)
      print(CONFIG, file=f)
      print("SAMPLE_NUM_PER_LABEL:", file=f)
      print(SAMPLE_NUM_PER_LABEL, file=f)
      print_summary(summary, f)


def train_and_create_result(p):
  _, x_support, x_test, _, y_support, y_test, _, y_support_value, y_test_value, input_shape = data_processing()
  # x_train, x_support, x_test, y_train, y_support, y_test, y_train_value, y_support_value, y_test_value, input_shape = data_processing()

  datasets = p.map(data_processing, range(CONFIG["num_models"]))

  for j in range(CONFIG["repeat"]):
    args = []
    for i in range(CONFIG["num_models"]):
      x_train, _, _, y_train, _, _, y_train_value, _, _, _ = datasets[i]
      ids = np.random.permutation(x_support.shape[0])
      ids = np.random.choice(ids, CONFIG["support_rate"])
      random_x_support = x_support[ids]
      random_y_support = y_support[ids]
      random_y_support_value = y_support_value[ids]
      x_train = np.vstack((x_train, random_x_support))
      y_train = np.vstack((y_train, random_y_support))
      y_train_value = np.hstack((y_train_value, random_y_support_value))
      args.append(
          [
              i,
              j,
              x_train,
              x_support,
              x_test,
              y_train,
              y_support,
              y_test,
              y_train_value,
              y_support_value,
              y_test_value,
              input_shape,
          ]
      )
    np.array(p.map(train, args))

  result = calc_ensemble_accuracy(
      x_test,
      y_test_value,
      p,
  )

  return result


def main():
  p = Pool(CONFIG["num_process"])
  results = []

  for i in range(CONFIG["experiment_count"]):
    print("-" * 200)
    print(
        "Experiment "
        + str(i + 1)
        + "/"
        + str(CONFIG["experiment_count"])
    )
    result = train_and_create_result(p)
    results.append(result)

  summary = create_summary(results)
  print_summary(summary)
  save_summary(summary)


if __name__ == "__main__":
  main()
