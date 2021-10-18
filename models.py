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


def build_fsl_dnn(input):
    x = Dense(121)(input)
    # x = Dense(242)(x)
    # x = Dense(60, activation="relu")(x)
    x = Dense(60, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(121, activation="softmax")(x)

    return x


def __build_fsl_cnn(input):
    x = Reshape((121, 1))(input)
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

    x_in = Flatten()(input)
    x = Add()(
        [
            x,
            x_in
        ]
    )
    # print(x.shape)
    # exit()

    return x


def build_fsl_cnn(input):
    # CNN 1
    cnn1_out = __cnn_block_for_nsl_kdd(input)

    # Addition 1
    addition1_in1 = Reshape((121,))(input)
    addition1_in2 = Reshape((121,))(cnn1_out)
    # addition1_in2 = Flatten()(l1_out)
    addition1_out = Add()([addition1_in1, addition1_in2])

    # Dense 1
    # dense1_out = Dense(121)(addition1_out)

    # Final Layour
    final_out = Reshape((121,))(addition1_out)

    return final_out


def build_fsl_attention(input):
    # Attention 1
    attention1_in = Reshape((121, 1))(input)
    # attention1_out = __attention_block(attention1_in, 121)
    attention1_out = __11_head_attention_block(attention1_in)

    # CNN 1
    cnn1_out = __cnn_block_for_nsl_kdd(attention1_out)

    # Addition 1
    addition1_in1 = Reshape((121,))(attention1_out)
    addition1_in2 = Reshape((121,))(cnn1_out)
    addition1_out = Add()([addition1_in1, addition1_in2])

    # Final Layour
    final_out = Reshape((121,))(addition1_out)
    # final_out = Reshape((121,))(attention1_out)

    return final_out


def __attention_block(input, dim):
    attention_in = Reshape((dim, 1))(input)
    x_attention = Attention(use_scale=True)([attention_in, attention_in])
    out = Add()([x_attention, attention_in])

    return out


def __11_head_attention_block(input):
    HEAD_NUM = 11
    attention_outputs = []
    split_input = Reshape((HEAD_NUM, 11))(input)

    for i in range(HEAD_NUM):
        attention_input = split_input[:, i, :]
        attention_output = __attention_block(attention_input, 11)
        reshaped_output = Reshape((11,))(attention_output)
        attention_outputs.append(reshaped_output)

    concat_output = Concatenate(axis=-1)(attention_outputs)

    return concat_output


def __cnn_block_for_nsl_kdd(input):
    # Reshape Input
    conv1_in = Reshape((11, 11, 1))(input)

    # Convolution 1
    conv1_out = __conv_block(conv1_in, 4)

    # Pooling 1
    pool1_out = __pool_block(conv1_out)

    # Convolution 2
    conv2_out = __conv_block(pool1_out, 16)

    # Pooling 2
    pool2_out = __pool_block(conv2_out)

    # Convolution 3
    conv3_out = __conv_block(pool2_out, 64)

    # Pooling 3
    pool3_out = __pool_block(conv3_out)

    # Convolution 4
    conv4_out = __conv_block(pool3_out, 121)

    return conv4_out


def __conv_block(input, channels):
    x = Conv2D(
        channels,
        3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
    )(input)
    # x = Conv2D(
    #     channels,
    #     3,
    #     strides=1,
    #     padding="same",
    #     kernel_initializer="he_normal",
    # )(x)
    return x


def __pool_block(input):
    x = MaxPooling2D(pool_size=2, strides=2)(input)
    return x


# Squeeze and Excitation
def __se_block(input, channels, r=6):
    # Squeeze
    x = GlobalAveragePooling2D()(input)
    # Excitation
    x = Dense(channels // r, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)
    x = Multiply()([input, x])
    return x
