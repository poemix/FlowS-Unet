import glob
import random
import tensorflow as tf
from utils import session, tfread_npy, tfread_tif


class FolderLoader(object):
    def __init__(self, data_root, mask_root, batch_size, height=256, width=256, transformer_fn=None, num_epochs=None,
                 data_suffix='npy', mask_suffix='tif', shuffle=True,
                 min_after_dequeue=25, allow_smaller_final_batch=False, num_threads=2, seed=None):
        self.data_root = data_root
        self.mask_root = mask_root
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.transformer_fn = transformer_fn
        self.num_epochs = num_epochs
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.shuffle = shuffle
        self.min_after_dequeue = min_after_dequeue
        self.allow_smaller_final_batch = allow_smaller_final_batch
        self.num_threads = num_threads
        self.seed = seed

        data_root = data_root.replace('\\', '/')
        data_root = data_root if data_root[-1] == '/' else '{0}/'.format(data_root)

        mask_root = mask_root.replace('\\', '/')
        mask_root = mask_root if mask_root[-1] == '/' else '{0}/'.format(mask_root)

        data_paths = glob.glob('{0}*.{1}'.format(data_root, data_suffix))
        data_paths = list(map(lambda x: x.replace('\\', '/'), data_paths))
        data_names = list(map(lambda x: x.split('/')[-1].split('.')[0], data_paths))
        mask_paths = list(map(lambda x: '{}{}.{}'.format(mask_root, x, mask_suffix), data_names))

        random.seed(seed)
        randnum = random.randint(0, 2018)
        random.seed(randnum)
        random.shuffle(data_paths)
        random.seed(randnum)
        random.shuffle(data_names)
        random.seed(randnum)
        random.shuffle(mask_paths)

        # print(data_paths)
        # print(data_names)
        # print(mask_paths)

        print('{}: create session!'.format(self.__class__.__name__))
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/cpu:0'):
                self.batch_ops, self.n_sample = self.folder_batch(data_paths=data_paths, mask_paths=mask_paths,
                                                                  data_names=data_names, batch_size=batch_size,
                                                                  transformer_fn=transformer_fn, num_epochs=num_epochs,
                                                                  shuffle=shuffle, min_after_dequeue=min_after_dequeue,
                                                                  allow_smaller_final_batch=allow_smaller_final_batch,
                                                                  num_threads=num_threads, seed=seed)
                if num_epochs is not None:
                    self.init = tf.local_variables_initializer()
        self.sess = session(graph=self.graph)
        if num_epochs is not None:
            self.sess.run(self.init)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def __len__(self):
        return self.n_sample

    def batch(self):
        return self.sess.run(self.batch_ops)

    @staticmethod
    def folder_batch(data_paths, mask_paths, data_names, batch_size, transformer_fn=None, num_epochs=None, shuffle=True,
                     min_after_dequeue=50, allow_smaller_final_batch=False, num_threads=2, seed=None, scope=None):
        with tf.name_scope(scope, 'folder_batch'):
            n_sample = len(data_paths)

            data_paths = tf.convert_to_tensor(data_paths, tf.string)
            mask_paths = tf.convert_to_tensor(mask_paths, tf.string)
            data_names = tf.convert_to_tensor(data_names, tf.string)

            data_path, mask_path, name = tf.train.slice_input_producer([data_paths, mask_paths, data_names],
                                                                       shuffle=shuffle,
                                                                       capacity=n_sample, seed=seed,
                                                                       num_epochs=num_epochs)

            data = tfread_npy(data_path)
            data = tf.reshape(data, tf.stack([256, 256, 8]))

            mask = tfread_tif(mask_path)
            mask = tf.reshape(mask, tf.stack([256, 256]))

            if transformer_fn is not None:
                data = transformer_fn(data)

            if shuffle:
                capacity = min_after_dequeue + (num_threads + 1) * batch_size
                data_batch, mask_batch, name_batch = tf.train.shuffle_batch([data, mask, name],
                                                                            batch_size=batch_size,
                                                                            capacity=capacity,
                                                                            min_after_dequeue=min_after_dequeue,
                                                                            num_threads=num_threads,
                                                                            allow_smaller_final_batch=allow_smaller_final_batch,
                                                                            seed=seed)
            else:
                capacity = (num_threads + 1) * batch_size
                data_batch, mask_batch, name_batch = tf.train.batch([data, mask, name],
                                                                    batch_size=batch_size,
                                                                    capacity=capacity,
                                                                    allow_smaller_final_batch=allow_smaller_final_batch)

            return [data_batch, mask_batch, name_batch], n_sample

    def __del__(self):
        print('{}: stop threads and close session!'.format(self.__class__.__name__))
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()


if __name__ == '__main__':
    loader = FolderLoader(data_root='E:/tianchi/dataset/s1/256v1/data', mask_root='E:/tianchi/dataset/s1/256v1/mask',
                          batch_size=16)

    for i in range(10):
        batch = loader.batch()
        print(batch[0].shape, batch[1].shape, list(map(lambda x: bytes.decode(x), batch[2])))
