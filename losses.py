from tensorflow.compat.v1.keras import backend as K
import numpy as np
from classifications import calc_centers_for_support_tensor
from constants import CONFIG
import tensorflow as tf


def center_loss(y_true, y_pred):
    SUPPORT_SIZE = int(CONFIG["test_sampling_method"][2:])

    indices = np.arange(CONFIG["batch_size"])
    train_indices = np.where(indices % (SUPPORT_SIZE + 1) == 0)[0]
    train_y_true = tf.gather(y_true, indices=train_indices)
    train_y_pred = tf.gather(y_pred, indices=train_indices)

    # support_indices = np.where(indices % (SUPPORT_SIZE + 1) != 0)[0]
    support_indices = np.arange(1, SUPPORT_SIZE + 1)
    support_y_true = tf.gather(y_true, indices=support_indices)
    support_y_pred = tf.gather(y_pred, indices=support_indices)

    centers = calc_centers_for_support_tensor(
        # support_y_true,
        K.concatenate([train_y_true, support_y_true], axis=0),
        # support_y_pred,
        K.concatenate([train_y_pred, support_y_pred], axis=0),
    )

    loss = K.variable(0.0)
    loss = tf.add(
        loss,
        1.0 * __center_loss(
            # train_y_true,
            K.concatenate([train_y_true, support_y_true], axis=0),
            # train_y_pred,
            K.concatenate([train_y_pred, support_y_pred], axis=0),
            centers
        )
    )
    loss = tf.add(
        loss,
        0.5 * __softmax_euclidean_loss(
            # train_y_true,
            K.concatenate([train_y_true, support_y_true], axis=0),
            # train_y_pred,
            K.concatenate([train_y_pred, support_y_pred], axis=0),
            centers
        )
    )
    # loss = tf.add(
    #     loss,
    #     0.1 * __softmax_cosine_loss(train_y_true, train_y_pred, centers)
    # )
    if CONFIG["experiment_id"] == "farther":
        loss = tf.add(
            loss,
            1.0 * __center_separate_loss(centers)
        )

    return loss


def __center_separate_loss(centers):
    distances = []
    # normal_distance = __euclidean(
    #     centers[0],
    #     K.zeros_like(centers[0])
    # )
    # distances.append(normal_distance)
    for label_i in range(CONFIG["num_classes"]):
        for label_j in range(CONFIG["num_classes"]):
            if label_i < label_j:
                center_i = centers[label_i]
                center_j = centers[label_j]
                distance = __euclidean(center_i, center_j)
                distances.append(distance)

    loss = K.variable(0.0)

    # average_distance = tf.reduce_mean(distances)

    for i in range(len(distances)):
        distance = distances[i]
        # distance = tf.multiply(-1., K.log(distance))
        # distance = tf.pow(distance, -2)
        distance = tf.pow(tf.subtract(distance, 2.5), 2)
        # distance = tf.log(distance)
        # distance = tf.multiply(-1., tf.pow(distance, 2))
        loss = tf.add(loss, distance)

    return loss


def __center_loss(y_true, y_pred, centers):
    y_true_value = K.argmax(y_true)

    loss = K.variable(0.0)
    for label in range(CONFIG["num_classes"]):
        center = centers[label]

        indices = tf.where(tf.equal(y_true_value, label))
        pred_per_class = tf.gather_nd(y_pred, indices=indices)
        diff = tf.subtract(pred_per_class, center)
        square_diff = K.pow(diff, 2)
        sum = K.sum(K.sum(square_diff, axis=-1), axis=-1)
        loss = tf.add(loss, sum)

    return loss


def __softmax_euclidean_loss(y_true, y_pred, centers):
    y_true_value = K.argmax(y_true)

    base_loss = K.variable(0.0)
    for label in range(CONFIG["num_classes"]):
        center = centers[label]
        indices = tf.where(tf.equal(y_true_value, label))
        pred_per_class = tf.gather_nd(y_pred, indices=indices)
        distances_per_class = __euclidean(pred_per_class, center)
        sum = K.sum(distances_per_class, axis=-1)
        base_loss = tf.add(base_loss, sum)

    base_distances = K.zeros_like(y_true_value, dtype='float32')
    for label in range(CONFIG["num_classes"]):
        center = centers[label]
        distances_with_all_classes = __euclidean(y_pred, center)
        exp_distances = K.exp(-distances_with_all_classes)
        base_distances = tf.add(base_distances, exp_distances)

    log_distances = K.log(base_distances)
    sum_on_batch = K.sum(log_distances)

    return base_loss + sum_on_batch


def __softmax_cosine_loss(y_true, y_pred, centers):
    y_true_value = K.argmax(y_true)

    base_loss = K.variable(0.0)
    for label in range(CONFIG["num_classes"]):
        center = centers[label]
        indices = tf.where(tf.equal(y_true_value, label))
        pred_per_class = tf.gather_nd(y_pred, indices=indices)
        distances_per_class = __cosine(pred_per_class, center)
        sum = K.sum(distances_per_class, axis=-1)
        base_loss = tf.add(base_loss, sum)

    base_distances = K.zeros_like(y_true_value, dtype='float32')
    for label in range(CONFIG["num_classes"]):
        center = centers[label]
        distances_with_all_classes = __cosine(y_pred, center)
        exp_distances = K.exp(-distances_with_all_classes)
        base_distances = tf.add(base_distances, exp_distances)

    log_distances = K.log(base_distances)
    sum_on_batch = K.sum(log_distances)

    return base_loss + sum_on_batch


def __euclidean(x, y):
    diff = tf.subtract(x, y)
    square_diff = K.pow(diff, 2)
    d = K.sqrt(K.sum(square_diff, axis=-1))
    return d


def __cosine(x, y):
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    c = K.mean(x * y, axis=-1)
    return -c
    # average = K.sum(c, axis=-1) / K.sum(K.ones_like(c), axis=-1)
    # return K.maximum(- c, - average * K.ones_like(c))
