from keras import backend as K
from keras.layers import (
    Add,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Embedding,
    Lambda,
    Concatenate
)

from constants import CONFIG


def build_base_cnn(inputs, input_target):
  x = Conv2D(4, (3, 3), padding="same")(inputs)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(16, (3, 3), padding="same")(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(64, (3, 3), padding="same")(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(121, (3, 3), padding="same")(x)
  x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
  x = Flatten()(x)
  f_inputs = Flatten()(inputs)
  ip1 = Add(name="ip1")([f_inputs, x])
  x = Dropout(0.5)(ip1)
  ip2 = Dense(CONFIG["num_classes"], activation="softmax", name="ip2")(x)

  centers = Embedding(CONFIG["num_classes"], CONFIG["img_rows"] * CONFIG["img_cols"])(input_target)
  l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name="l2_loss")([ip1, centers])

  return ip2, l2_loss


def build_fsl_cnn(inputs):
  x = Conv2D(4, (3, 3), padding="same")(inputs)
  x = MaxPooling2D(strides=(2, 2))(x)
  x = Conv2D(12, (3, 3), padding="same")(x)
  x = MaxPooling2D(strides=(2, 2))(x)
  x = Conv2D(36, (3, 3), padding="same")(x)
  x = MaxPooling2D(strides=(2, 2))(x)
  x = Conv2D(121, (3, 3), padding="same")(x)
  # x = MaxPooling2D(strides=(2, 2), padding="same")(x)
  # x = Dropout(0.2)(x)
  x = Flatten()(x)

  x2 = Flatten()(inputs)
  # x2 = Dense(32)(x2)
  x = Add()([x, x2])
  # x = Dropout(0.25)(x)
  # x = Dense(64)(x)
  # x = Dropout(0.2)(x)
  # x = Dense(CONFIG["output_dim"])(x)

  return x


def build_fsl_dnn(inputs):
  x = Flatten()(inputs)
  x = Dense(121)(x)
  # x = Dropout(0.2)(x)
  x = Dense(64)(x)
  # x = Dropout(0.2)(x)
  x = Dense(32)(x)
  x = Dropout(0.2)(x)
  x = Dense(CONFIG["output_dim"])(x)

  return x
