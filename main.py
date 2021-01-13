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

from constants import CONFIG, LABELS
from classifications import calc_ensemble_accuracy, calc_distance, load_distances
from data_processing import data_processing
from callbacks import Histories
from models import build_fsl_cnn, build_fsl_dnn
from summary import create_summary, print_summary, save_summary
from losses import center_loss
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import multiprocessing as mp
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


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
      batch_size=CONFIG["batch_size"],  # + index * 5,
      epochs=CONFIG["epochs"] if j >= 0 else 1,
      verbose=False,
      callbacks=[
          histories
      ],
      shuffle=CONFIG["shuffle"],
  )

  file_name = "./temp/model_" + str(index) + "_" + str(j) + ".h5"
  if os.path.isfile(file_name):
    os.remove(file_name)
  model.save_weights(file_name)

  K.clear_session()


def train_and_create_result(p, e_i):
  _, x_support, x_test, _, y_support, y_test, _, y_support_value, y_test_value, input_shape = data_processing(
      [
          None,
          "zero",
      ]
  )
  # x_train, x_support, x_test, y_train, y_support, y_test, y_train_value, y_support_value, y_test_value, input_shape = data_processing()
  # ids = np.random.permutation(x_support.shape[0])
  # ids = np.random.choice(ids, CONFIG["support_rate"])
  # random_x_support = x_support[ids]
  # random_y_support = y_support[ids]
  # random_y_support_value = y_support_value[ids]
  # x_train = np.vstack((x_train, random_x_support))
  # y_train = np.vstack((y_train, random_y_support))
  # y_train_value = np.hstack((y_train_value, random_y_support_value))

  args = [
      [i,
       [
           "a",
           "d",
           "f",
       ][i % 3]
       ] for i in range(CONFIG["num_models"])
  ]

  datasets = p.map(data_processing, args)

  for j in range(CONFIG["repeat"]):
    args = []
    for i in range(CONFIG["num_models"]):
      x_train, _, _, y_train, _, _, y_train_value, _, _, _ = datasets[i]
      support_ids = np.random.permutation(x_support.shape[0])
      # support_ids = np.random.choice(support_ids, CONFIG["support_rate"])
      support_ids = np.tile(
          support_ids,
          # (CONFIG["support_rate"] // len(x_support)) * int(i / 1 + 1)
          5,
      )
      random_x_support = x_support[support_ids]
      random_y_support = y_support[support_ids]
      random_y_support_value = y_support_value[support_ids]

      train_ids = np.random.permutation(x_train.shape[0])
      random_x_train = x_train[train_ids]
      random_y_train = y_train[train_ids]
      random_y_train_value = y_train_value[train_ids]

      x_train = np.vstack((random_x_train, random_x_support))
      y_train = np.vstack((random_y_train, random_y_support))
      y_train_value = np.hstack((random_y_train_value, random_y_support_value))

      del support_ids, random_x_support, random_y_support, random_y_support_value, train_ids, random_x_train, random_y_train, random_y_train_value, train_normal_ids, x_train_normal, y_train_normal, y_train_value_normal

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
    p.map(train, args)

  result = calc_ensemble_accuracy(
      x_test,
      y_test_value,
      p,
      e_i,
  )

  return result


def main():
  p = mp.get_context('spawn').Pool(CONFIG["num_process"])
  results = []

  for i in range(CONFIG["experiment_count"]):
    print("-" * 200)
    print(
        "Experiment "
        + str(i + 1)
        + "/"
        + str(CONFIG["experiment_count"])
    )
    result = train_and_create_result(p, i)
    results.append(result)

    pf = platform.system()
    if pf == 'Darwin':
      p.close()
      p.join()

      p = mp.get_context('spawn').Pool(CONFIG["num_process"])

  summary = create_summary(results)
  print_summary(summary)
  save_summary(summary)


def comparison_train_dataset():
  p = mp.get_context('spawn').Pool(CONFIG["num_process"])
  sampling_methods = [
      "a",
      "b",
      "c",
      "d",
      "e",
      "f"
  ]
  results = {}
  args = [
      [
          e_i,
          "zero"
      ] for e_i in range(CONFIG["experiment_count"] * CONFIG["experiment_count"])
  ]
  support_datasets = p.map(data_processing, args)
  for sampling_method in sampling_methods:
    results[sampling_method] = {
        "accuracy": [],
    }
    for label in LABELS:
      results[sampling_method][label] = []

    args = [
        [
            e_i,
            sampling_method,
        ] for e_i in range(CONFIG["experiment_count"] * CONFIG["experiment_count"])
    ]
    datasets = p.map(data_processing, args)

    for i in range(CONFIG["experiment_count"]):
      print("-" * 200)
      print(
          "Experiment "
          + str(i + 1)
          + "/"
          + str(CONFIG["experiment_count"])
          + " " + sampling_method
      )

      _, _, x_test, _, _, y_test, _, _, y_test_value, input_shape = support_datasets[i]
      args = []
      for j in range(CONFIG["experiment_count"]):
        x_train, _, _, y_train, _, _, y_train_value, _, _, _ = datasets[
            CONFIG["experiment_count"] * i + j
        ]
        _, x_support, _, _, y_support, _, _, y_support_value, _, _ = support_datasets[
            CONFIG["experiment_count"] * i + j
        ]
        support_ids = np.random.permutation(x_support.shape[0])
        support_ids = np.tile(
            support_ids,
            5,
        )
        random_x_support = x_support[support_ids]
        random_y_support = y_support[support_ids]
        random_y_support_value = y_support_value[support_ids]

        train_ids = np.random.permutation(x_train.shape[0])
        random_x_train = x_train[train_ids]
        random_y_train = y_train[train_ids]
        random_y_train_value = y_train_value[train_ids]

        x_train = np.vstack((random_x_train, random_x_support))
        y_train = np.vstack((random_y_train, random_y_support))
        y_train_value = np.hstack(
            (
                random_y_train_value,
                random_y_support_value
            )
        )

        args.append(
            [
                j,
                0,
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
      p.map(train, args)
      distances = np.array(
          p.map(load_distances, range(CONFIG["experiment_count"])))
      for distance in distances:
        pred = np.argmin(distance[-1], axis=1)
        report = classification_report(
            y_test_value,
            pred,
            output_dict=True,
            target_names=LABELS
        )
        for type in report:
          if type == "accuracy":
            results[sampling_method][type].append(report[type])
          elif type in LABELS:
            results[sampling_method][type].append(report[type]["recall"])

  for sampling_method in sampling_methods:
    print(sampling_method)
    for type in results[sampling_method]:
      mean = np.mean(results[sampling_method][type]) * 100
      std = np.std(results[sampling_method][type]) * 100
      min = np.min(results[sampling_method][type]) * 100
      max = np.max(results[sampling_method][type]) * 100
      print(
          type
          + "\t\t"
          + "$" + "{:.02f}".format(mean) + "\pm" + "{:.02f}".format(std)+"$"
          + " min: " + "{:.02f}".format(min)
          + " max: " + "{:.02f}".format(max),
      )


def comparison_test_dataset():
  p = mp.get_context('spawn').Pool(CONFIG["num_process"])
  sampling_methods = [
      # "a",
      "b",
      "c",
      "d",
      "e",
      "f"
  ]
  results = {}
  args = [
      [
          e_i,
          "zero"
      ] for e_i in range(CONFIG["experiment_count"] * CONFIG["experiment_count"])
  ]
  datasets = p.map(data_processing, args)
  for sampling_method in sampling_methods:
    results[sampling_method] = {
        "accuracy": [],
    }
    for label in LABELS:
      results[sampling_method][label] = []

    args = [
        [
            e_i,
            sampling_method,
        ] for e_i in range(CONFIG["experiment_count"] * CONFIG["experiment_count"])
    ]
    support_datasets = p.map(data_processing, args)

    for i in range(CONFIG["experiment_count"]):
      print("-" * 200)
      print(
          "Experiment "
          + str(i + 1)
          + "/"
          + str(CONFIG["experiment_count"])
          + " " + sampling_method
      )

      _, _, x_test, _, _, y_test, _, _, y_test_value, input_shape = datasets[i]
      args = []
      for j in range(CONFIG["experiment_count"]):
        _, x_support, _, _, y_support, _, _, y_support_value, _, _ = support_datasets[
            CONFIG["experiment_count"] * i + j
        ]
        x_train, _, _, y_train, _, _, y_train_value, _, _, _ = datasets[
            CONFIG["experiment_count"] * i + j
        ]
        support_ids = np.random.permutation(x_support.shape[0])
        support_ids = np.tile(
            support_ids,
            5,
        )
        random_x_support = x_support[support_ids]
        random_y_support = y_support[support_ids]
        random_y_support_value = y_support_value[support_ids]

        train_ids = np.random.permutation(x_train.shape[0])
        random_x_train = x_train[train_ids]
        random_y_train = y_train[train_ids]
        random_y_train_value = y_train_value[train_ids]

        x_train = np.vstack((random_x_train, random_x_support))
        y_train = np.vstack((random_y_train, random_y_support))
        y_train_value = np.hstack(
            (
                random_y_train_value,
                random_y_support_value
            )
        )

        args.append(
            [
                j,
                0,
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
      p.map(train, args)
      distances = np.array(
          p.map(load_distances, range(CONFIG["experiment_count"])))
      for distance in distances:
        pred = np.argmin(distance[-1], axis=1)
        report = classification_report(
            y_test_value,
            pred,
            output_dict=True,
            target_names=LABELS
        )
        for type in report:
          if type == "accuracy":
            results[sampling_method][type].append(report[type])
          elif type in LABELS:
            results[sampling_method][type].append(report[type]["recall"])

  for sampling_method in sampling_methods:
    print(sampling_method)
    for type in results[sampling_method]:
      mean = np.mean(results[sampling_method][type]) * 100
      std = np.std(results[sampling_method][type]) * 100
      min = np.min(results[sampling_method][type]) * 100
      max = np.max(results[sampling_method][type]) * 100
      print(
          type
          + "\t\t"
          + "$" + "{:.02f}".format(mean) + "\pm" + "{:.02f}".format(std)+"$"
          + " min: " + "{:.02f}".format(min)
          + " max: " + "{:.02f}".format(max),
      )


if __name__ == "__main__":
  main()
  # comparison_train_dataset()
  # comparison_test_dataset()
