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
    # Layer 1
    l1_conv_in = Reshape((11, 11, 1))(inputs)
    l1_conv_out = __conv_block(l1_conv_in, 4)
    l1_out = __pool_block(l1_conv_out)

    # Layer 2
    l2_conv_out = __conv_block(l1_out, 16)
    l2_out = __pool_block(l2_conv_out)

    # Layer 3
    l3_conv_out = __conv_block(l2_out, 64)
    l3_out = __pool_block(l3_conv_out)

    # Layer 4
    l4_out = __conv_block(l3_out, CONFIG["output_dim"])

    # Layer 5
    l5_in1 = Reshape((121,))(inputs)
    l5_in2 = Reshape((121,))(l4_out)
    # l5_in2 = Flatten()(l1_out)
    l5_out = Add()([l5_in1, l5_in2])

    # Layer 6
    # l6_out = Dense(121)(l5_out)

    # Final Layour
    final_out = Reshape((121,))(l5_out)

    return final_out


def build_fsl_attention(inputs):
    # Layer 1
    l1_in = Reshape((121, 1))(inputs)
    l1_out = __attention_block(l1_in)

    # Layer 2
    l2_conv_in = Reshape((11, 11, 1))(l1_out)
    l2_conv_out = __conv_block(l2_conv_in, 4)
    l2_out = __pool_block(l2_conv_out)

    # Layer 3
    l3_conv_out = __conv_block(l2_out, 16)
    l3_out = __pool_block(l3_conv_out)

    # Layer 4
    l4_conv_out = __conv_block(l3_out, 64)
    l4_out = __pool_block(l4_conv_out)

    # Layer 5
    l5_out = __conv_block(l4_out, CONFIG["output_dim"])

    # Layer 6
    l6_in1 = Reshape((121,))(l1_out)
    l6_in2 = Reshape((121,))(l5_out)
    l6_out = Add()([l6_in1, l6_in2])

    # Final Layour
    final_out = Reshape((121,))(l6_out)

    return final_out


def __attention_block(inputs):
    x_attention = Attention(use_scale=True)([inputs, inputs])
    x = Add()([x_attention, inputs])

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
