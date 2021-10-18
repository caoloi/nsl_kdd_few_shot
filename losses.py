from tensorflow.compat.v1.keras import backend as K
import numpy as np
from classifications import calc_centers


def center_loss(x_support, y_support_value, model):
    def c_loss(y_true, y_pred):
        centers = calc_centers(x_support, y_support_value, model)

        loss = K.variable(0.0)
        loss = loss + 1.0 * __center_loss(y_true, y_pred, centers)
        loss = loss + 1.0 * __softmax_euclidean_loss(y_true, y_pred, centers)
        # loss = loss + 1.0 * __softmax_cosine_loss(y_true, y_pred, centers)

        return loss

    return c_loss


def __center_loss(y_true, y_pred, centers):
    loss = K.sum(
        K.pow(
            y_pred - K.dot(
                y_true[:, :5],
                K.variable(
                    centers
                ),
            ),
            2,
        ),
        axis=-1
    )

    return loss


def __softmax_euclidean_loss(y_true, y_pred, centers):
    loss = __euclidean(
        y_pred,
        K.dot(
            y_true[:, :5],
            K.variable(
                centers
            )
        )
    )
    tmp_loss = K.variable(0.0)
    for center in centers:
        tmp_loss = tmp_loss + K.exp(
            -__euclidean(
                y_pred,
                K.dot(
                    K.ones_like(
                        y_true[:, :1]
                    ),
                    K.variable(
                        np.array(
                            [
                                center
                            ]
                        )
                    ),
                )
            )
        )
    loss = loss + K.log(tmp_loss)

    return loss


def __softmax_cosine_loss(y_true, y_pred, centers):
    loss = __cosine(
        y_pred,
        K.dot(
            y_true[:, :5],
            K.variable(
                centers
            )
        )
    )
    tmp_loss = K.variable(0.0)
    for center in centers:
        tmp_loss = tmp_loss + K.exp(
            -__cosine(
                y_pred,
                K.dot(
                    K.ones_like(
                        y_true[:, :1]
                    ),
                    K.variable(
                        np.array(
                            [
                                center
                            ]
                        )
                    ),
                )
            )
        )
        loss = loss + K.log(tmp_loss)

    return loss


def __euclidean(x, y):
    d = K.sqrt(K.sum(K.square(x - y), axis=-1))
#   d = K.sum(d, axis=-1)
    return d
    # average = K.sum(d, axis=-1) / K.sum(K.ones_like(d), axis=-1)
    # return K.maximum(d, average * K.ones_like(d))


def __cosine(x, y):
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    c = K.sum(x * y, axis=-1)
    # c = K.sum(c)
    return - c
    # average = K.sum(c, axis=-1) / K.sum(K.ones_like(c), axis=-1)
    # return K.maximum(- c, - average * K.ones_like(c))
