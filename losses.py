from keras import backend as K
import numpy as np
from classifications import calc_centers, calc_centers_2


def center_loss(x_support, y_support, y_support_value, model):
  def c_loss(y_true, y_pred):
    centers = calc_centers(x_support, y_support_value, model)
    # train_centers = calc_centers_2(y_pred, y_true[:, :5])

    loss1 = K.sum(
        K.square(
            y_pred - K.dot(
                y_true[:, :5],
                K.variable(
                    centers
                ),
            )
        ),
        axis=-1
    )
    # loss2 = - K.sum(K.square(y_pred - K.batch_dot(K.ones_like(y_true[:, :5]) - y_true[:, :5], K.variable(centers), axes=(1, 0))), axis=-1)

    # loss3 = __euclidean(
    #     y_pred,
    #     K.dot(
    #         y_true[:, :5],
    #         K.variable(
    #             centers
    #         )
    #     )
    # )
    # tmp_loss3 = K.variable(0.0)
    # for center in centers:
    #   tmp_loss3 = tmp_loss3 + K.exp(
    #       -__euclidean(
    #           y_pred,
    #           K.dot(
    #               K.ones_like(
    #                   y_true[:, :1]
    #               ),
    #               K.variable(
    #                   np.array(
    #                       [
    #                           center
    #                       ]
    #                   )
    #               ),
    #           )
    #       )
    #   )
    # loss3 = loss3 + K.log(tmp_loss3)

    loss4 = __cosine(
        y_pred,
        K.dot(
            y_true[:, :5],
            K.variable(
                centers
            )
        )
    )
    tmp_loss4 = K.variable(0.0)
    for center in centers:
      tmp_loss4 = tmp_loss4 + K.exp(
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
    loss4 = loss4 + K.log(tmp_loss4)

    # support_pred = model.predict(x_support)

    # loss5 = __cosine(K.variable(support_pred), K.variable(np.dot(y_support, train_centers)))
    # tmp_loss5 = K.variable(0.0)
    # for center in train_centers:
    #   tmp_loss5 += K.exp(-__cosine(K.variable(support_pred), K.variable([center for _ in range(support_pred.shape[0])])))
    # loss5 += K.log(tmp_loss5)

    # loss6 = K.sum(K.square(K.variable(support_pred - np.dot(y_support, train_centers))), axis=-1)

    loss = 1.0 * loss1 + 1.0 * loss4
    # loss = loss5 + loss6

    return loss

  return c_loss


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
