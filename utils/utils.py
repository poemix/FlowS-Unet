

import cv2
import numpy as np
import tensorflow as tf
from experiment.hyperparams import HyperParams as hp


def session(graph=None, allow_soft_placement=hp.allow_soft_placement, log_device_placement=hp.log_device_placement,
            allow_growth=hp.allow_growth):
    """return a session with simple config"""
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)


def tfread_npy(path):
    def fn(p):
        return np.load(bytes.decode(p)).astype(np.float32)
    return tf.py_func(fn, [path], tf.float32)


def tfread_tif(path):
    def fn(p):
        return cv2.imread(bytes.decode(p), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return tf.py_func(fn, [path], tf.float32)
