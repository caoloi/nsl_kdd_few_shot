from keras import backend as K
from keras.layers import (
    Add,
    Dropout,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Embedding,
    Lambda,
    Concatenate,
    Reshape,
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

  centers = Embedding(
      CONFIG["num_classes"], CONFIG["img_rows"] * CONFIG["img_cols"])(input_target)
  l2_loss = Lambda(lambda x: K.sum(
      K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name="l2_loss")([ip1, centers])

  return ip2, l2_loss


def build_fsl_cnn(inputs):
  # x = Conv2D(4, 3, strides=1, padding="same")(inputs)
  # x = AveragePooling2D(pool_size=2, strides=1)(x)
  # x = Conv2D(12, 3, strides=1, padding="same")(x)
  # x = AveragePooling2D(pool_size=2, strides=2)(x)
  # x = Conv2D(36, 2, strides=1, padding="valid")(x)
  # x = MaxPooling2D(pool_size=2, strides=2)(x)
  # x = Conv2D(121, 2, strides=1, padding="same")(x)
  # x = MaxPooling2D(pool_size=2, strides=2)(x)
  # x = Flatten()(x)

  x2 = Conv2D(4, 3, strides=1, padding="same")(inputs)
  # x2 = AveragePooling2D(pool_size=2, strides=1)(x2)
  x2 = MaxPooling2D(pool_size=2, strides=2)(x2)
  x2 = Conv2D(12, 3, strides=1, padding="same")(x2)
  x2 = MaxPooling2D(pool_size=2, strides=2)(x2)
  x2 = Conv2D(36, 3, strides=1, padding="same")(x2)
  x2 = MaxPooling2D(pool_size=2, strides=2)(x2)
  x2 = Conv2D(121, 3, strides=1, padding="same")(x2)
  x2 = Flatten()(x2)
  # x2 = Dense(121)(x2)

  x_in = Flatten()(inputs)
  x = Add()(
    [
      # x,
      x2,
      x_in
    ]
  )

  x_in = Flatten()(inputs)
  x = Add()([x, x_in])

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
