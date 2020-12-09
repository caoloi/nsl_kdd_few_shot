import platform
pf = platform.system()
if pf == 'Darwin':
  import plaidml.keras
  import os
  plaidml.keras.install_backend()
  os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
  os.environ["PLAIDML_EXPERIMENTAL"] = "1"
  # os.environ["PLAIDML_DEVICE_IDS"] = "opencl_amd_gfx1010.0"
else:
  import os
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#   from keras import backend as K
#   import tensorflow as tf
#   config = tf.compat.v1.ConfigProto()
#   config.gpu_options.per_process_gpu_memory_fraction = 0.1
#   config.gpu_options.allow_growth = True
#   sess = tf.compat.v1.Session(config=config)
#   K.set_session(sess)
#   K.clear_session()

from constants import CONFIG
from classifications import calc_ensemble_accuracy
from data_processing import data_processing
from callbacks import Histories
from models import build_fsl_cnn, build_fsl_dnn
from losses import center_loss
from keras.optimizers import Adam, SGD, RMSprop, Nadam, Adamax
from keras.models import Model, model_from_json, load_model
from keras.layers import (
    Input
)
import numpy as np
from multiprocessing import Pool, cpu_count, set_start_method
from time import sleep


def train(args):
  import platform
  pf = platform.system()
  if pf != 'Darwin':
    from keras import backend as K
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.07
    # config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
    K.clear_session()

  index, x_train, x_support, x_test, y_train, y_support, _, y_train_value, y_support_value, y_test_value, input_shape = args

  # sleep(5 * (index % CONFIG["num_process"]))

  print("Setting up Model " + str(index + 1) + "/" + str(CONFIG["num_models"]))

  input = Input(shape=input_shape)
  output = build_fsl_cnn(
      input
  ) if CONFIG["model_type"] == "cnn" else build_fsl_dnn(input)
  model = Model(inputs=input, outputs=output)
  model.compile(
      # optimizer=RMSprop(),
      # optimizer=SGD(lr=0.1),
      # optimizer=Nadam(),
      optimizer=Adam(),
      # optimizer=Adamax(),
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
      index
  )

  model.fit(
      x_train,
      expanded_y_train,
      batch_size=CONFIG["batch_size"] + index * 10,
      epochs=CONFIG["epochs"],
      verbose=False,
      callbacks=[
          histories
      ]
  )


def load_models(index):
  print("Load Model " + str(index + 1) + "/" + str(CONFIG["num_models"]))

  models = [
      # load_model(
      #     "./temp/model_" + str(index) + "_epoch_" + str(j) + ".h5",
      #     compile=False
      # )
      np.load(
        "./temp/model_" + str(index) + "_epoch_" + str(j) + ".npy"
      )
      for j in range(CONFIG["epochs"])
  ]
  return models


def main():
  _, x_support, x_test, _, y_support, y_test, _, y_support_value, y_test_value, input_shape = data_processing()
  # x_train, x_support, x_test, y_train, y_support, y_test, y_train_value, y_support_value, y_test_value, input_shape = data_processing()
  p = Pool(CONFIG["num_process"])
  datasets = p.map(data_processing, range(CONFIG["num_models"]))

  args = []
  for i in range(CONFIG["num_models"]):
    x_train, _, _, y_train, _, _, y_train_value, _, _, _ = datasets[i]
    ids = np.random.permutation(x_support.shape[0])
    ids = np.random.choice(ids, CONFIG["support_rate"])
    # ids = [
    #     i % x_support.shape[0]
    #     for i in range(
    #         int(
    #             x_train.shape[0] * CONFIG["support_rate"]
    #         )
    #     )
    # ]
    random_x_support = x_support[ids]
    random_y_support = y_support[ids]
    random_y_support_value = y_support_value[ids]
    x_train = np.vstack((x_train, random_x_support))
    y_train = np.vstack((y_train, random_y_support))
    y_train_value = np.hstack((y_train_value, random_y_support_value))
    args.append(
        [
            i,
            x_train,
            x_support,
            x_test,
            y_train,
            y_support,
            y_test,
            y_train_value,
            y_support_value,
            y_test_value,
            input_shape
        ]
    )
  np.array(p.map(train, args))

  print("-" * 200)

  models = np.array(p.map(load_models, range(CONFIG["num_models"])))

  # x_supports = results[:, 0]
  # y_supports = results[:, 1]

  calc_ensemble_accuracy(
    x_test,
    y_test_value,
    x_support,
    y_support_value,
    models,
    p,
  )


if __name__ == "__main__":
  set_start_method("spawn", True)
  main()
