import tensorflow as tf
import numpy as np


def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice, axis=0)

def crossentropy_loss(y_true, y_pred):
    ce = tf.keras.losses.CategoricalCrossentropy()
    return ce(y_true, y_pred)   

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    pt = tf.exp(-ce)
    fl = (1 - pt) ** gamma * ce
    return tf.reduce_mean(fl)

def combined_loss(alpha=0.3, beta=0.7):
    def loss_function(y_true, y_pred):
        #fl = focal_loss(y_true, y_pred, gamma=gamma)
        dl = dice_loss(y_true, y_pred)
        ce = crossentropy_loss(y_true, y_pred)
        return alpha * ce + beta * dl
    return loss_function

def create_iou_metric(y_true, y_pred, num_classes):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)

    ious = []
    for i in range(1, num_classes):  # начинаем с 1, игнорируем класс 0
        y_true_i = tf.cast(tf.equal(y_true, i), tf.float32)
        y_pred_i = tf.cast(tf.equal(y_pred, i), tf.float32)

        intersection = tf.reduce_sum(y_true_i * y_pred_i)
        union = tf.reduce_sum(y_true_i) + tf.reduce_sum(y_pred_i) - intersection

        iou = tf.cond(
            tf.equal(union, 0.0),
            lambda: tf.constant(0.0, dtype=tf.float32),
            lambda: intersection / union
        )
        ious.append(iou)
    mean_iou = tf.reduce_mean(tf.stack(ious))
    return mean_iou

def mean_iou_metric(num_classes):
    def mean_iou(y_true, y_pred):
        return create_iou_metric(y_true, y_pred, num_classes)
    mean_iou.__name__ = 'mean_iou'
    return mean_iou
