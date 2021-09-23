from tensorflow.keras.layers import (
    Add,
    Dropout,
    Conv1D,
    Conv2D,
    MaxPooling1D,
    MaxPooling2D,
    AveragePooling1D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Flatten,
    Dense,
    Multiply,
    Embedding,
    Lambda,
    Concatenate,
    Reshape,
    BatchNormalization,
    Activation,
    Attention,
    AdditiveAttention,
)

from constants import CONFIG


def build_fsl_dnn(inputs):
    x = Dense(121)(inputs)
    # x = Dense(242)(x)
    # x = Dense(60, activation="relu")(x)
    x = Dense(60, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(121, activation="softmax")(x)

    return x


def __build_fsl_cnn(inputs):
    x = Reshape((121, 1))(inputs)
    # print(x.shape)
    x = Conv1D(4, 2, padding="valid")(x)
    # print(x.shape)
    x = MaxPooling1D(pool_size=2)(x)
    # print(x.shape)
    x = Conv1D(8, 8, padding="same")(x)
    # print(x.shape)
    x = MaxPooling1D(pool_size=2)(x)
    # print(x.shape)
    x = Conv1D(16, 8, padding="same")(x)
    # print(x.shape)
    x = MaxPooling1D(pool_size=2)(x)
    # print(x.shape)
    x = Conv1D(32, 8, padding="same")(x)
    # print(x.shape)
    x = MaxPooling1D(pool_size=3)(x)
    # print(x.shape)
    x = Conv1D(64, 2, padding="valid")(x)
    # print(x.shape)
    x = MaxPooling1D(pool_size=2)(x)
    # print(x.shape)
    x = Conv1D(128, 8, padding="same")(x)
    # print(x.shape)
    x = MaxPooling1D(pool_size=2)(x)
    # print(x.shape)
    x = Flatten()(x)
    # print(x.shape)
    x = Dense(121)(x)
    # print(x.shape)

    x_in = Flatten()(inputs)
    x = Add()(
        [
            x,
            x_in
        ]
    )
    # print(x.shape)
    # exit()

    return x


def build_fsl_cnn(inputs):
    x = Reshape((121, 1))(inputs)
    x_attention = __multi_head_attention_block(x)

    x = Reshape((11, 11, 1))(x_attention)
    x = __conv_block(x, 4)
    x = __pool_block(x)

    x = __conv_block(x, 16)
    x = __pool_block(x)

    x = __conv_block(x, 64)
    x = __pool_block(x)

    x = __conv_block(x, CONFIG["output_dim"])
    # x = __conv_block(x, 121)

    x = Flatten()(x)

    x_in = Flatten()(x_attention)
    x = Add()(
        [
            x,
            x_in
        ]
    )

    x = Dense(CONFIG["output_dim"])(x)

    return x


def __multi_head_attention_block(inputs):
    # x_attention = AdditiveAttention(
    #     use_scale=True,
    # )([inputs, inputs])
    x_attention = Attention(
        use_scale=True,
    )([inputs, inputs])
    x = Add()(
        [
            x_attention,
            inputs,
        ]
    )

    return x


def __conv_block(inputs, channels):
    x = Conv2D(
        channels,
        3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    # x = Conv2D(
    #     channels,
    #     3,
    #     strides=1,
    #     padding="same",
    #     kernel_initializer="he_normal",
    # )(x)
    return x


def __pool_block(inputs):
    x = MaxPooling2D(pool_size=2, strides=2)(inputs)
    return x


# Squeeze and Excitation
def __se_block(inputs, channels, r=6):
    # Squeeze
    x = GlobalAveragePooling2D()(inputs)
    # Excitation
    x = Dense(channels // r, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)
    x = Multiply()([inputs, x])
    return x
