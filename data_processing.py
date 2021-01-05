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
    SAMPLE_NUM_PER_LABEL,
    SERVICE_VALUES,
    FLAG_VALUES,
    PROTOCOL_TYPE_VALUES,
    COLUMNS
)
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import boxcox
pd.options.mode.chained_assignment = None  # default="warn" | Disable warnings


def data_processing(index=None):
  if index is not None:
    print("Load Dataset " + str(index + 1) + "/" + str(CONFIG["num_models"]))

  x_train, x_support, x_test, y_train, y_support, y_test = __load_data(index)

  if image_data_format() == "channels_first" and CONFIG["model_type"] == "cnn":
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
  elif CONFIG["model_type"] == "cnn":
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
  else:
    input_shape = (121,)

  # Maintain single value ground truth labels for center loss inputs
  # Because Embedding layer only accept index as inputs instead of one-hot vector
  y_train_value = y_train
  y_support_value = y_support
  y_test_value = y_test

  # convert class vectors to binary class matrices
  y_train = to_categorical(y_train, CONFIG["num_classes"])
  y_support = to_categorical(y_support, CONFIG["num_classes"])
  y_test = to_categorical(y_test, CONFIG["num_classes"])

  return x_train, x_support, x_test, y_train, y_support, y_test, y_train_value, y_support_value, y_test_value, input_shape


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


def __load_data(index):
  if CONFIG["dataset"] == "kdd":
    x_train, x_support, x_test, y_train, y_support, y_test = __kdd_encoding(index)
  else:
    x_train, x_support, x_test, y_train, y_support, y_test = __kdd_encoding(index)

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


def __kdd_encoding(index):
  train_df, test_df = __load_kdd_dataset()

  train_df, test_df = __numerical_processing(train_df, test_df)

  # train_df = __smote_processing(train_df)
  train_df = __resample_processing(
      train_df,
      index,
      balanced=True,
  )
  x_train, y_train = __label_to_num_processing(train_df)
  support_df = __resample_processing(
      test_df,
      index,
      balanced=True,
      type="test",
  )
  # support_df = __smote_processing(
  #     support_df,
  #     r=CONFIG["smote_rate"],
  #     type="test",
  # )
  x_support, y_support = __label_to_num_processing(support_df)

  test_df = __resample_processing(
      test_df,
      index,
      balanced=False,
  )
  x_test, y_test = __label_to_num_processing(test_df)

  if CONFIG["model_type"] == "cnn":
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
  else:
    x_train = x_train.to_numpy()
    x_support = x_support.to_numpy()
    x_test = x_test.to_numpy()
  y_train = y_train.to_numpy()
  y_support = y_support.to_numpy()
  y_test = y_test.to_numpy()

  return x_train, x_support, x_test, y_train, y_support, y_test


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

      # boxcox
      # if train_df[c].min() != train_df[c].max():
      #   train_df[c] = train_df[c] + 1
      #   train_df[c] = boxcox(train_df[c], lmbda=0.5)
      # if test_df[c].min() != test_df[c].max():
      #   test_df[c] = test_df[c] + 1
      #   test_df[c] = boxcox(test_df[c], lmbda=0.5)

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


def __resample_processing(df, index, balanced, type="train"):
  if index is None or type == "test":
    index = 1
  else:
    # index = index % 3 + 1
    index = 1

  if balanced:
    df_per_category = {}
    for label in ENTRY_TYPE:
      for minor_label in ENTRY_TYPE[label]:
        df_per_category[minor_label] = df[df["label"] == minor_label[:-1]]

    df_list = []
    for label in SAMPLE_NUM_PER_LABEL:
      if SAMPLE_NUM_PER_LABEL[label][type] >= 0:
        if type == "test":
          # sorted_df = df_per_category[label].sort_values("difficulty")
          # temp_df = sorted_df.tail(10)
          # temp_df = sorted_df.head(5).append(sorted_df.tail(5))
          # temp_df = sorted_df.tail(np.max([SAMPLE_NUM_PER_LABEL[label][type], len(sorted_df) // 4]))
          # temp_df = sorted_df[(len(sorted_df) // 4):(3 * (len(sorted_df) // 4))]
          # temp_df = df_per_category[label].sort_values("difficulty").head(len(df_per_category[label]) // 2)
          # samples = temp_df.sample(
          #     n=SAMPLE_NUM_PER_LABEL[label][type],
          #     replace=SAMPLE_NUM_PER_LABEL[label][type] > len(temp_df)
          # )
          samples = df_per_category[label].sample(
              n=SAMPLE_NUM_PER_LABEL[label][type],
              replace=SAMPLE_NUM_PER_LABEL[label][type] > len(
                  df_per_category[label]
              )
          )
        else:
          # print(label)
          # print(len(df_per_category[label]))
          # temp_df = df_per_category[label].sort_values("difficulty").head(len(df_per_category[label]) // 2)
          # temp_df = df_per_category[label].sort_values(
          #     "difficulty"
          # )[
          #     (
          #         len(
          #             df_per_category[label]
          #         ) // 4
          #     ):
          # ]
          # print(len(temp_df))
          temp_df = df_per_category[label]
          samples = temp_df.sample(
              n=index * SAMPLE_NUM_PER_LABEL[label][type],
              replace=index * SAMPLE_NUM_PER_LABEL[label][type] > len(
                  temp_df
              )
          )
        df_list.append(samples)
      # samples = df_per_category[label].sample(n=int(np.log(len(df_per_category[label]) + 1)))
      # df_list.append(samples)
    df = pd.concat(df_list, ignore_index=True)

    # Assign x (inputs) and y (outputs) of the network
    df = df.sample(frac=1)

  return df


def __label_to_num_processing(df):
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

  # print(df["label"].value_counts())
  y = df["label"]
  # x = df.drop(columns="label")
  x = df.drop(
      columns=[
          "label",
          "difficulty",
      ]
  )

  return x, y


def __smote_processing(df, r=1, type="train"):
  classes = []
  for k in SAMPLE_NUM_PER_LABEL:
    if SAMPLE_NUM_PER_LABEL[k][type] > 0:
      classes.append(k[:-1])
  df = df[df["label"].isin(classes)]
  x = df.drop(columns="label")
  y = df["label"]
  # print('Original dataset samples per class {}'.format(Counter(y)))

  strategy = Counter(y)
  for k in strategy:
    strategy[k] = np.max(
        [
            strategy[k],
            (1 if k == "normal" else r) *
            SAMPLE_NUM_PER_LABEL[k + "."][type]
        ]
    )
  sm = SMOTENC(
      sampling_strategy=strategy,
      categorical_features=[
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15,
          16,
          17,
          18,
          19,
          20,
          21,
          22,
          23,
          24,
          25,
          26,
          27,
          28,
          29,
          30,
          31,
          32,
          33,
          34,
          35,
          36,
          37,
          38,
          39,
          40,
          41,
          42,
          43,
          44,
          45,
          46,
          47,
          48,
          49,
          50,
          51,
          52,
          53,
          54,
          55,
          56,
          57,
          58,
          59,
          60,
          61,
          62,
          63,
          64,
          65,
          66,
          67,
          68,
          69,
          70,
          71,
          72,
          73,
          74,
          75,
          76,
          77,
          78,
          79,
          80,
          81,
          82,
          83,
          84,
          87,
          92,
          100,
          101,
          121,
      ],
      k_neighbors=(2 if type == "test" else 5),
      n_jobs=-1
  )
  x, y = sm.fit_resample(x, y)
  # print('Resampled dataset samples per class {}'.format(Counter(y)))
  df = pd.concat([y, x], axis=1)
  return df


def __rank_gauss(df):
  for col in df.columns:
    if col != "label" and col != "difficulty":
      transformer = QuantileTransformer(
          n_quantiles=100,
          random_state=0,
          output_distribution='normal'
      )
      vec_len = len(df[col].values)
      raw_vec = df[col].values.reshape(vec_len, 1)
      transformer.fit(raw_vec)

      # 変換後のデータで各列を置換
      df[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]

  return df
