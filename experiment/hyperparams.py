import os


class HyperParams(object):
    pro_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")).replace('\\', '/')
    # random seed
    seed = 2018

    # optimizer
    learning_rate = 0.0001
    stepsize = 5000
    lr_decay = 0.1

    num_epoch = 15
    snapshot_iter = 100
    display = 1

    # save model
    output_dir = '{}/model/{}'
    snapshot_infix = ''
    snapshot_prefix = 'flows-unet'

    # data loader
    shuffle = True
    min_after_dequeue = 64
    allow_smaller_final_batch = False
    num_threads = 2
    batch_size = 4

    # session config
    allow_soft_placement = False
    log_device_placement = False
    allow_growth = True
