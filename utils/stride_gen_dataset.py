import os
import cv2
import numpy as np
from osgeo import gdal
from experiment.hyperparams import HyperParams as hp


def stride_gen_npy(file_2015='E:/tianchi/dataset/preliminary/2015_s1.tif',
                   file_2017='E:/tianchi/dataset/preliminary/2017_s1.tif',
                   file_mask='{}/data/mask/mask_s1_final.tif'.format(hp.pro_path),
                   data_save_path="E:/tianchi/dataset/s1/256v1/data",
                   mask_save_path="E:/tianchi/dataset/s1/256v1/mask",
                   window=(256, 256), stride=(150, 150), ch=8,
                   data_suffix='npy', mask_suffix='tif'):
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    HEIGHT, WIDTH = window
    nYStride, nXStride = stride

    im_2015 = gdal.Open(file_2015)
    im_2017 = gdal.Open(file_2017)
    im_mask = cv2.imread(file_mask, cv2.IMREAD_UNCHANGED)

    nXSize = im_2015.RasterXSize
    nYSize = im_2017.RasterYSize

    n_width = int(np.floor((nXSize - WIDTH) / nXStride) + 1)
    n_height = int(np.floor((nYSize - HEIGHT) / nYStride) + 1)

    print(n_width, n_height)
    print("im_2015.shape", (im_2015.RasterYSize, im_2015.RasterXSize))
    print("im_2017.shape", (im_2017.RasterYSize, im_2017.RasterXSize))
    print("im_label.shape", im_mask.shape)

    for i in range(n_height):
        for j in range(n_width):
            data_name = "s1_{}-{}_{}-{}_{}-{}.{}".format(i, j, HEIGHT, WIDTH, nYStride, nXStride, data_suffix)
            mask_name = "s1_{}-{}_{}-{}_{}-{}.{}".format(i, j, HEIGHT, WIDTH, nYStride, nXStride, mask_suffix)

            arr_2015 = im_2015.ReadAsArray(j * nXStride, i * nYStride, WIDTH, HEIGHT)
            arr_2015 = arr_2015.transpose([1, 2, 0])

            arr_2017 = im_2017.ReadAsArray(j * nXStride, i * nYStride, WIDTH, HEIGHT)
            arr_2017 = arr_2017.transpose([1, 2, 0])

            concat_img = np.concatenate((arr_2017[:, :, :(ch // 2)], arr_2015[:, :, :(ch // 2)]), axis=2)
            print(data_name)
            print(concat_img.dtype)
            np.save('{}/{}'.format(data_save_path, data_name), concat_img)

            print(mask_name)
            print(im_mask.dtype, im_mask.shape)
            cv2.imwrite(
                '{}/{}'.format(mask_save_path, mask_name),
                im_mask[i * nYStride: i * nYStride + HEIGHT, j * nXStride: j * nXStride + WIDTH]
            )


def stride_gen_rgb(file_2015='E:/tianchi/dataset/preliminary/2015_s1.tif',
                   file_2017='E:/tianchi/dataset/preliminary/2017_s1.tif',
                   file_mask='{}/data/mask/mask_s1_final.tif'.format(hp.pro_path),
                   data_save_path="E:/tianchi/dataset/s1/256v1/data",
                   mask_save_path="E:/tianchi/dataset/s1/256v1/mask",
                   window=(256, 256), stride=(150, 150), ch=8,
                   data_suffix='npy', mask_suffix='tif'):
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    HEIGHT, WIDTH = window
    nYStride, nXStride = stride

    im_2015 = gdal.Open(file_2015)
    im_2017 = gdal.Open(file_2017)
    im_mask = cv2.imread(file_mask, cv2.IMREAD_UNCHANGED)

    nXSize = im_2015.RasterXSize
    nYSize = im_2017.RasterYSize

    n_width = int(np.floor((nXSize - WIDTH) / nXStride) + 1)
    n_height = int(np.floor((nYSize - HEIGHT) / nYStride) + 1)

    print(n_width, n_height)
    print("im_2015.shape", (im_2015.RasterYSize, im_2015.RasterXSize))
    print("im_2017.shape", (im_2017.RasterYSize, im_2017.RasterXSize))
    print("im_label.shape", im_mask.shape)

    for i in range(n_height):
        for j in range(n_width):
            data_name = "s1_{}-{}_{}-{}_{}-{}.{}".format(i, j, HEIGHT, WIDTH, nYStride, nXStride, data_suffix)
            mask_name = "s1_{}-{}_{}-{}_{}-{}.{}".format(i, j, HEIGHT, WIDTH, nYStride, nXStride, mask_suffix)

            arr_2015 = im_2015.ReadAsArray(j * nXStride, i * nYStride, WIDTH, HEIGHT)
            arr_2015 = arr_2015.transpose([1, 2, 0])

            arr_2017 = im_2017.ReadAsArray(j * nXStride, i * nYStride, WIDTH, HEIGHT)
            arr_2017 = arr_2017.transpose([1, 2, 0])

            concat_img = np.concatenate((arr_2017[:, :, :(ch // 2)], arr_2015[:, :, :(ch // 2)]), axis=2)
            print(data_name)
            print(concat_img.dtype)
            np.save('{}/{}'.format(data_save_path, data_name), concat_img)

            print(mask_name)
            print(im_mask.dtype, im_mask.shape)
            cv2.imwrite(
                '{}/{}'.format(mask_save_path, mask_name),
                im_mask[i * nYStride: i * nYStride + HEIGHT, j * nXStride: j * nXStride + WIDTH]
            )


if __name__ == '__main__':
    # stride_gen_npy()
    stride_gen_rgb()
    pass
