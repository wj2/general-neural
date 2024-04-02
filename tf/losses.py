
import tensorflow as tf

def mse_nanloss(label, prediction):
    nan_mask = tf.math.logical_not(tf.math.is_nan(label))
    label = tf.boolean_mask(label, nan_mask)
    prediction = tf.boolean_mask(prediction, nan_mask)
    mult = tf.square(prediction - label)
    mse_nanloss = tf.reduce_mean(mult)
    return mse_nanloss


def binary_crossentropy_nan(label, prediction):
    nan_mask = tf.math.logical_not(tf.math.is_nan(label))
    label = tf.boolean_mask(label, nan_mask)
    prediction = tf.boolean_mask(prediction, nan_mask)
    bcel = tf.keras.losses.binary_crossentropy(label, prediction)
    return bcel
