import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import OrderedDict


class FlowSUnet(object):
    def __init__(self, data, mask, keep_prob, is_training, n_class=2, depth=4, features_root=32):
        self.data = data
        self.mask = mask
        self.keep_prob = keep_prob
        self.is_training = is_training

        self.n_class = n_class
        self.depth = depth
        self.features_root = features_root

        self.layers = dict(data=data, mask=mask)

        self.setup()

    def setup(self):
        with tf.variable_scope('FlowS-Unet'):
            dw_hs = OrderedDict()
            outputs = OrderedDict()
            # down layers
            inputs = self.data
            for i in range(0, self.depth):
                features = 2 ** i * self.features_root
                conv1 = slim.conv2d(inputs, features, kernel_size=[3, 3], stride=1)
                bn1 = slim.batch_norm(conv1)
                conv2 = slim.conv2d(bn1, features, kernel_size=[3, 3], stride=1)
                bn2 = slim.batch_norm(conv2)
                dw_hs[i] = bn2

                if i < self.depth - 1:
                    pool = slim.max_pool2d(bn2, kernel_size=[2, 2], stride=2)

                    # update inputs
                    inputs = pool

            # up layers
            inputs1 = dw_hs[self.depth - 1]
            inputs2 = None
            for j in range(self.depth - 2, -1, -1):
                features = 2 ** (j + 1) * self.features_root
                deconv1 = slim.conv2d_transpose(inputs1, features // 2, kernel_size=[2, 2], stride=2)

                if j == self.depth - 2:
                    concat = tf.concat([dw_hs[j], deconv1], axis=3)
                else:
                    deconv2 = slim.conv2d_transpose(inputs2, 1, kernel_size=[2, 2], stride=2)
                    concat = tf.concat([dw_hs[j], deconv1, deconv2], axis=3)
                    concat = tf.nn.dropout(concat, keep_prob=self.keep_prob)
                conv1 = slim.conv2d(concat, features // 2, kernel_size=[3, 3], stride=1)
                conv2 = slim.conv2d(conv1, features // 2, kernel_size=[3, 3], stride=1)
                conv3 = slim.conv2d(conv2, self.n_class, kernel_size=[1, 1], stride=1, padding='VALID')

                # update inputs1, 2
                inputs1 = conv2
                inputs2 = conv3
                outputs[j] = conv3

            self.layers['outputs'] = outputs


if __name__ == '__main__':
    data = tf.placeholder(tf.float32, [None, 256, 256, 4])
    mask = tf.placeholder(tf.float32, [None, 256, 256, 2])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    net = FlowSUnet(data=data, mask=mask, keep_prob=keep_prob, is_training=is_training)

