from imblearn.over_sampling import SMOTENC, SMOTE, SVMSMOTE
from collections import Counter
import pandas as pd
import numpy as np
from keras.backend import image_data_format
from keras.utils import to_categorical
from constants import (
    CONFIG,
    LABEL_TO_NUM,
    FULL_FEATURES,
    ENTRY_TYPE,
    TRAIN_SAMLE_NUM_PER_LABEL,
    TEST_SAMLE_NUM_PER_LABEL,
    SAMPLE_NUM_PER_LABEL,
    SERVICE_VALUES,
    FLAG_VALUES,
    PROTOCOL_TYPE_VALUES,
    COLUMNS,
)
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import boxcox
pd.options.mode.chained_assignment = None  # default="warn" | Disable warnings
from copy import copy


def create_csv():
  train_df, test_df = __load_kdd_dataset()
  train_df, test_df = __numerical_processing(train_df, test_df)

  for index in range(CONFIG["num_models"]):
    train_df.to_csv("./temp/train_df_" + str(index) + ".csv")

  train_df.to_csv("./temp/train_df.csv")
  test_df.to_csv("./temp/test_df.csv")


def data_processing(args):
  index, method = args

  if index is not None:
    print(
        "Load Dataset "
        + str(index + 1)
        + "/"
        + str(CONFIG["num_models"])
        + " "
        + method
    )

  x_train, y_train, y_train_value, input_shape = train_data_processing(args)
  x_support, y_support, y_support_value, _ = support_data_processing(args)
  x_test, y_test, y_test_value, _, y_test_orig = test_data_processing(args)

  return x_train, x_support, x_test, y_train, y_support, y_test, y_train_value, y_support_value, y_test_value, y_test_orig, input_shape


def t_sne_data_processing():
  train_df, test_df = __load_kdd_dataset()

  train_df, test_df = __numerical_processing(train_df, test_df)

  x_train, _ = __resample_processing(
      train_df,
      balanced=False,
  )
  x_support, _ = __resample_processing(
      test_df,
      balanced=True,
      type="test",
  )
  x_test, _ = __resample_processing(
      test_df,
      balanced=False,
  )

  y_train = x_train.T.index.values
  x_train = x_train.T.to_numpy()
  y_support = x_support.T.index.values
  x_support = x_support.T.to_numpy()
  y_test = x_test.T.index.values
  x_test = x_test.T.to_numpy()

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


def train_data_processing(args):
  index, method = args

  if index is None:
    train_df = pd.read_csv("./temp/train_df.csv", index_col=0)
  else:
    train_df = pd.read_csv(
        "./temp/train_df_" + str(index) + ".csv",
        index_col=0
    )
  train_df = train_df.sample(frac=1)

  train_df = __resample_processing(
      train_df,
      index,
      method,
      balanced=True,
  )
  x_train, y_train, _ = __label_to_num_processing(train_df)

  if CONFIG["model_type"] == "cnn":
    x_train = np.array(x_train).reshape(
        x_train.shape[0],
        CONFIG["img_rows"],
        CONFIG["img_cols"]
    )
  else:
    x_train = x_train.to_numpy()
  y_train = y_train.to_numpy()

  if image_data_format() == "channels_first" and CONFIG["model_type"] == "cnn":
    x_train = x_train.reshape(
        x_train.shape[0], 1, CONFIG["img_rows"], CONFIG["img_cols"]
    )
    input_shape = (1, CONFIG["img_rows"], CONFIG["img_cols"])
  elif CONFIG["model_type"] == "cnn":
    x_train = x_train.reshape(
        x_train.shape[0], CONFIG["img_rows"], CONFIG["img_cols"], 1
    )
    input_shape = (CONFIG["img_rows"], CONFIG["img_cols"], 1)
  else:
    input_shape = (121,)

  y_train_value = y_train
  y_train = to_categorical(y_train, CONFIG["num_classes"])

  return x_train, y_train, y_train_value, input_shape


def support_data_processing(args):
  index, method = args

  test_df = pd.read_csv("./temp/test_df.csv", index_col=0)
  test_df = test_df.sample(frac=1)

  support_df = __resample_processing(
      test_df,
      index,
      method,
      balanced=True,
      type="test",
  )
  x_support, y_support, _ = __label_to_num_processing(support_df)

  if CONFIG["model_type"] == "cnn":
    x_support = np.array(x_support).reshape(
        x_support.shape[0],
        CONFIG["img_rows"],
        CONFIG["img_cols"]
    )
  else:
    x_support = x_support.to_numpy()
  y_support = y_support.to_numpy()

  if image_data_format() == "channels_first" and CONFIG["model_type"] == "cnn":
    x_support = x_support.reshape(
        x_support.shape[0], 1, CONFIG["img_rows"], CONFIG["img_cols"]
    )
    input_shape = (1, CONFIG["img_rows"], CONFIG["img_cols"])
  elif CONFIG["model_type"] == "cnn":
    x_support = x_support.reshape(
        x_support.shape[0], CONFIG["img_rows"], CONFIG["img_cols"], 1
    )
    input_shape = (CONFIG["img_rows"], CONFIG["img_cols"], 1)
  else:
    input_shape = (121,)

  y_support_value = y_support
  y_support = to_categorical(y_support, CONFIG["num_classes"])

  return x_support, y_support, y_support_value, input_shape


def test_data_processing(args):
  index, method = args

  test_df = pd.read_csv("./temp/test_df.csv", index_col=0)

  test_df = __resample_processing(
      test_df,
      index,
      method,
      balanced=False,
  )
  x_test, y_test, y_test_orig = __label_to_num_processing(test_df)

  if CONFIG["model_type"] == "cnn":
    x_test = np.array(x_test).reshape(
        x_test.shape[0],
        CONFIG["img_rows"],
        CONFIG["img_cols"]
    )
  else:
    x_test = x_test.to_numpy()
  y_test = y_test.to_numpy()

  if image_data_format() == "channels_first" and CONFIG["model_type"] == "cnn":
    x_test = x_test.reshape(
        x_test.shape[0], 1, CONFIG["img_rows"], CONFIG["img_cols"]
    )
    input_shape = (1, CONFIG["img_rows"], CONFIG["img_cols"])
  elif CONFIG["model_type"] == "cnn":
    x_test = x_test.reshape(
        x_test.shape[0], CONFIG["img_rows"], CONFIG["img_cols"], 1
    )
    input_shape = (CONFIG["img_rows"], CONFIG["img_cols"], 1)
  else:
    input_shape = (121,)

  y_test_value = y_test
  y_test = to_categorical(y_test, CONFIG["num_classes"])

  return x_test, y_test, y_test_value, input_shape, y_test_orig


def __numerical_processing(train_df, test_df):
  train_df = __column_processing(train_df)
  test_df = __column_processing(test_df)

  for c in COLUMNS:
    if c != "label" and c != "difficulty":
      train_df[c] = train_df[c].astype(float)
      test_df[c] = test_df[c].astype(float)

      # x = log(x + 1)
      if train_df[c].min() != train_df[c].max():
        train_df[c] = np.log(train_df[c] + 1)
      if test_df[c].min() != test_df[c].max():
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
          # "difficulty",
          "num_outbound_cmds",
          "service",
          "protocol_type",
          "flag",
      ]
  )

  return df


def __resample_processing(df, index, method, balanced, type="train"):
  if index is None or type == "test":
    index = 1
  else:
    # index = index % 3 + 1
    index = 1

  if balanced:
    df_per_category = {}
    for label in ENTRY_TYPE:
      for minor_label in ENTRY_TYPE[label]:
        df_per_category[minor_label] = df[df["label"] == minor_label]

    df_list = []
    ii = 0
    for label in SAMPLE_NUM_PER_LABEL:
      if SAMPLE_NUM_PER_LABEL[label][type] >= 0:
        if type == "test":
          if CONFIG["test_sampling_method"] == "zero":
            samples = __tile_samples(
                df_per_category[label],
                TEST_SAMLE_NUM_PER_LABEL[method][ii])
          else:
            samples = __tile_samples(
                df_per_category[label],
                TEST_SAMLE_NUM_PER_LABEL[CONFIG["test_sampling_method"]][ii]
            )
        else:
          if CONFIG["train_sampling_method"] == "zero":
            samples = __tile_samples(
                df_per_category[label],
                index * TRAIN_SAMLE_NUM_PER_LABEL[method][ii]
            )
          else:
            samples = __tile_samples(
                df_per_category[label],
                index *
                TRAIN_SAMLE_NUM_PER_LABEL[CONFIG["train_sampling_method"]][ii]
            )
        df_list.append(samples)
      ii += 1
    df = pd.concat(df_list, ignore_index=True)
    df = df.sample(frac=1)

  return df


def __label_to_num_processing(df):
  y_orig = copy(df["label"])

  # Replace connexion type string with an int (also works with NSL)
  for label in ENTRY_TYPE:
    for i in range(len(ENTRY_TYPE[label])):
      df["label"] = df["label"].replace(
          ENTRY_TYPE[label][i],
          LABEL_TO_NUM[label]
      )

  # print(df["label"].value_counts())
  y = df["label"]
  # x = df.drop(columns="label")
  x = df.drop(
      columns=[
          "label",
          "difficulty",
      ]
  )

  return x, y, y_orig


def __tile_samples(df, num):
  ids = []
  id = 0
  df_len = len(df)
  while len(ids) < num:
    ids.append(id)
    id += 1
    id %= df_len
  return df.iloc[ids]
