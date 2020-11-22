from constants import (
    CONFIG,
    LABEL_TO_NUM,
    FULL_FEATURES,
    ENTRY_TYPE,
    SAMPLE_NUM_PER_LABEL,
    SERVICE_VALUES,
    FLAG_VALUES,
    PROTOCOL_TYPE_VALUES,
    COLUMNS
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils import to_categorical
from keras.backend import image_data_format
import numpy as np
from multiprocessing import cpu_count
from time import sleep
import pandas as pd
pd.options.mode.chained_assignment = None  # default="warn" | Disable warnings


def data_processing(index=None):
  if index is not None:
    sleep((index % cpu_count()) * 0.1)
    print("Load Dataset " + str(index + 1) + "/" + str(CONFIG["num_models"]))

  x_train, x_support, x_test, y_train, y_support, y_test = __load_data()

  if image_data_format() == "channels_first":
    x_train = x_train.reshape(
        x_train.shape[0],
        1,
        CONFIG["img_rows"],
        CONFIG["img_cols"]
    )
    x_support = x_support.reshape(
        x_support.shape[0],
        1,
        CONFIG["img_rows"],
        CONFIG["img_cols"]
    )
    x_test = x_test.reshape(
        x_test.shape[0],
        1,
        CONFIG["img_rows"],
        CONFIG["img_cols"]
    )

    input_shape = (
        1,
        CONFIG["img_rows"],
        CONFIG["img_cols"]
    )
  else:
    x_train = x_train.reshape(
        x_train.shape[0],
        CONFIG["img_rows"],
        CONFIG["img_cols"],
        1
    )
    x_support = x_support.reshape(
        x_support.shape[0],
        CONFIG["img_rows"],
        CONFIG["img_cols"],
        1
    )
    x_test = x_test.reshape(
        x_test.shape[0],
        CONFIG["img_rows"],
        CONFIG["img_cols"],
        1
    )

    input_shape = (
        CONFIG["img_rows"],
        CONFIG["img_cols"],
        1
    )

  # Maintain single value ground truth labels for center loss inputs
  # Because Embedding layer only accept index as inputs instead of one-hot vector
  y_train_value = y_train
  y_support_value = y_support
  y_test_value = y_test

  # convert class vectors to binary class matrices
  y_train = to_categorical(y_train, CONFIG["num_classes"])
  y_support = to_categorical(y_support, CONFIG["num_classes"])
  y_test = to_categorical(y_test, CONFIG["num_classes"])

  if index is None:
    return x_train, x_support, x_test, y_train, y_support, y_test, y_train_value, y_support_value, y_test_value, input_shape
  else:
    return index, x_train, x_support, x_test, y_train, y_support, y_test, y_train_value, y_support_value, y_test_value, input_shape


def t_sne_data_processing():
  train_df, test_df = __load_kdd_dataset()

  train_df, test_df = __numerical_processing(train_df, test_df)

  x_train, _ = __resample_processing(
      train_df,
      balanced=False,
      supported=False,
  )
  x_support, _ = __resample_processing(
      test_df,
      balanced=True,
      supported=True,
      type="test",
  )
  x_test, _ = __resample_processing(
      test_df,
      balanced=False,
      supported=False,
  )

  y_train = x_train.T.index.values
  x_train = x_train.T.to_numpy()
  y_support = x_support.T.index.values
  x_support = x_support.T.to_numpy()
  y_test = x_test.T.index.values
  x_test = x_test.T.to_numpy()

  return x_train, x_support, x_test, y_train, y_support, y_test


def __load_data():
  if CONFIG["dataset"] == "kdd":
    x_train, x_support, x_test, y_train, y_support, y_test = __kdd_encoding()
  else:
    x_train, x_support, x_test, y_train, y_support, y_test = __kdd_encoding()

  return x_train, x_support, x_test, y_train, y_support, y_test


def __load_kdd_dataset():
  # ***** DATA PATH *****
  data_path = "./data/"
  # 10% kdd99 train + kdd99 test datasets
  if CONFIG["train_data"] == "10_percent":
    train_data_path = data_path + "kddcup_traindata_10_percent.csv"
    test_data_path = data_path + "kddcup_testdata_corrected.csv"
  # Full kdd99 train + kdd99 test datasets
  elif CONFIG["train_data"] == "normal":
    train_data_path = data_path + "kddcup_traindata.csv"
    test_data_path = data_path + "kddcup_testdata_corrected.csv"
  else:
    # Full NSL kdd train + NSL kdd test datasets
    if CONFIG["train_data"] == "train+":
      train_data_path = data_path + "KDDTrain+.csv"
    # 20% NSL kdd train + NSL kdd test datasets
    elif CONFIG["train_data"] == "train+20":
      train_data_path = data_path + "KDDTrain+_20Percent.csv"
    else:
      train_data_path = data_path + "kddcup_traindata.csv"
    test_data_path = data_path + "KDDTest+.csv"
    # test_data_path = data_path+"KDDTest-21.csv"
    # FULL_FEATURES.append("difficulty")

  # Load csv data into dataframes and name the feature
  train_df = pd.read_csv(train_data_path, names=FULL_FEATURES)
  test_df = pd.read_csv(test_data_path, names=FULL_FEATURES)

  return train_df, test_df


def __kdd_encoding():
  train_df, test_df = __load_kdd_dataset()

  train_df, test_df = __numerical_processing(train_df, test_df)

  x_train, y_train = __resample_processing(
      train_df,
      balanced=True,
      supported=True,
  )
  # x_train_support, y_train_support = __resample_processing(
  #     train_df,
  #     balanced=True,
  #     supported=False,
  #     nums=[
  #         6,
  #         6,
  #         6,
  #         6,
  #         6,
  #     ],
  # )
  x_support, y_support = __resample_processing(
      # x_test_support, y_test_support = __resample_processing(
      test_df,
      balanced=True,
      supported=True,
      type="test",
  )
  # x_support, y_support = np.array(np.concatenate([x_train_support, x_test_support])), np.array(np.concatenate([y_train_support, y_test_support]))
  # p = np.random.permutation(len(x_support))
  # x_support = x_support[p]
  # y_support = y_support[p]
  # x_support, y_support = __resample_processing(
  #     pd.concat([train_df, test_df], ignore_index=True),
  #     balanced=True,
  #     supported=True,
  #     nums=[
  #         52,
  #         52,
  #         52,
  #         52,
  #         52,
  #     ],
  # )
  x_test, y_test = __resample_processing(
      test_df,
      balanced=False,
      supported=False,
  )

  x_train = np.array(x_train).reshape(
      x_train.shape[0],
      CONFIG["img_rows"],
      CONFIG["img_cols"]
  )
  x_support = np.array(x_support).reshape(
      x_support.shape[0],
      CONFIG["img_rows"],
      CONFIG["img_cols"]
  )
  x_test = np.array(x_test).reshape(
      x_test.shape[0],
      CONFIG["img_rows"],
      CONFIG["img_cols"]
  )
  y_train = y_train.to_numpy()
  y_support = y_support.to_numpy()
  y_test = y_test.to_numpy()

  return x_train, x_support, x_test, y_train, y_support, y_test


def __numerical_processing(train_df, test_df):
  train_df = __column_processing(train_df)
  test_df = __column_processing(test_df)

  for c in COLUMNS:
    if c != "label":
      train_df[c] = train_df[c].astype(float)
      test_df[c] = test_df[c].astype(float)

      # x = log(x + 1)
      train_df[c] = np.log(train_df[c] + 1)
      test_df[c] = np.log(test_df[c] + 1)

      # scale to range [0, 1]
      min = np.min([train_df[c].min(), test_df[c].min()])
      max = np.max([train_df[c].max(), test_df[c].max()])
      if min != max:
        train_df[c] = (train_df[c] - min) / (max - min)
        test_df[c] = (test_df[c] - min) / (max - min)

  # sort
  train_df = train_df[COLUMNS]
  test_df = test_df[COLUMNS]

  return train_df, test_df


def __column_processing(df):
  df = df[FULL_FEATURES]

  # Replace String features with ints
  for i in range(len(SERVICE_VALUES)):
    df[SERVICE_VALUES[i]] = list(
        map(
            lambda x: 1 if x == SERVICE_VALUES[i] else 0,
            df["service"]
        )
    )

  for i in range(len(PROTOCOL_TYPE_VALUES)):
    df[PROTOCOL_TYPE_VALUES[i]] = list(
        map(
            lambda x: 1 if x == PROTOCOL_TYPE_VALUES[i] else 0,
            df["protocol_type"]
        )
    )

  for i in range(len(FLAG_VALUES)):
    df[FLAG_VALUES[i]] = list(
        map(
            lambda x: 1 if x == FLAG_VALUES[i] else 0,
            df["flag"]
        )
    )

  # drop columns
  df = df.drop(
      columns=[
          "difficulty",
          "num_outbound_cmds",
          "service",
          "protocol_type",
          "flag",
      ]
  )

  return df


def __resample_processing(df, balanced, supported, type="train", nums=[]):
  if supported:
    df_per_category = {}
    for label in ENTRY_TYPE:
      for minor_label in ENTRY_TYPE[label]:
        df_per_category[minor_label] = df[df["label"] == minor_label[:-1]]

    df_list = []
    for label in SAMPLE_NUM_PER_LABEL:
      if SAMPLE_NUM_PER_LABEL[label][type == "test"] >= 0:
        samples = df_per_category[label].sample(
            n=SAMPLE_NUM_PER_LABEL[label][type == "test"],
            replace=SAMPLE_NUM_PER_LABEL[label][type == "test"] > len(df_per_category[label])
        )
        df_list.append(samples)
      # samples = df_per_category[label].sample(n=int(np.log(len(df_per_category[label]) + 1)))
      # df_list.append(samples)
    df = pd.concat(df_list, ignore_index=True)

    # Assign x (inputs) and y (outputs) of the network
    df = df.sample(frac=1)

  # Replace connexion type string with an int (also works with NSL)
  for label in ENTRY_TYPE:
    for i in range(len(ENTRY_TYPE[label])):
      df["label"] = df["label"].replace(
        [
            ENTRY_TYPE[label][i],
            ENTRY_TYPE[label][i][:-1]
        ],
          LABEL_TO_NUM[label]
      )

  if balanced and (not supported):
    df_per_category = []
    for i in range(CONFIG["num_classes"]):
      df_per_category.append(df[df["label"] == i])

    df_list = []
    for i in range(CONFIG["num_classes"]):
      if nums[i % 5] == 0:
        df_list.append(df_per_category[i])
      else:
        if len(df_per_category[i]) > 0:
          samples = df_per_category[i].sample(n=nums[i % 5], replace=nums[i % 5] > len(df_per_category[i]))
          df_list.append(samples)
    balanced_df = pd.concat(df_list, ignore_index=True)

    # Assign x (inputs) and y (outputs) of the network
    balanced_df = balanced_df.sample(frac=1)

    y = balanced_df["label"]
    x = balanced_df.drop(columns="label")
  else:
    y = df["label"]
    x = df.drop(columns="label")

  return x, y
