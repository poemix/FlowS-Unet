

import os
import cv2
import numpy as np


DATA_DIR = "E:\\tianchi\\rgb"

IM_ROWS = 5106
IM_COLS = 15106
ROI_SIZE = 503
ALPHA = 0.5  # 1是不透明，0是全透明


def on_mouse(event, x, y, flags, params):
    img, points = params['img'], params['points']
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

    if event == cv2.EVENT_RBUTTONDOWN:
        if len(points) != 0:
            points.pop()

    temp = img.copy()
    if len(points) > 2:
        cv2.fillPoly(temp, [np.array(points)], (0, 0, 255))

    for i in range(len(points)):
        cv2.circle(temp, points[i], 1, (0, 0, 255))

    cv2.circle(temp, (x, y), 3, (0, 255, 0))
    cv2.imshow("2017", temp)


def label_img(src2015, src2017, pred2, save_path):
    masked = src2017.copy()
    label = None
    if os.path.exists(save_path):
        label = cv2.imread(save_path, 0)
        masked[label > 128] = masked[label > 128] * (1 - ALPHA) + np.array([(0, 0, 255)]) * ALPHA

    c = 0

    # show src2015
    cv2.namedWindow("2015", 0)
    cv2.imshow("2015", src2015)

    # show cada
    # pre = src2017.copy()
    # cv2.namedWindow("pred", 0)
    # pre[pred > 128] = np.array([(0, 0, 255)])
    # cv2.imshow("pred", pre)

    # show cada
    pre2 = src2017.copy()
    cv2.namedWindow("pred2", 0)
    pre2[pred2 > 128] = np.array([(0, 0, 255)])
    cv2.imshow("pred2", pre2)

    while c not in [110, 112, 27]:
        temp = masked.copy()

        cv2.namedWindow("2017", 0)
        points = []
        cv2.setMouseCallback("2017", on_mouse, {'img': temp, 'points': points})
        cv2.imshow("2017", masked)

        c = cv2.waitKey(0)

        if c == 115:
            if label is None:
                # 标签不存在
                l = np.zeros((256, 256), dtype=np.uint8)
                cv2.imwrite(save_path, l)
                if len(points) > 0:
                    cv2.fillPoly(masked, [np.array(points)], (0, 0, 255))
                    cv2.fillPoly(l, [np.array(points)], (255, 255, 255))
                    cv2.imwrite(save_path, l)
                label = l
            else:
                # 标签存在基于标签标注
                if len(points) > 0:
                    cv2.fillPoly(masked, [np.array(points)], (0, 0, 255))
                    cv2.fillPoly(label, [np.array(points)], (255, 255, 255))
                    cv2.imwrite(save_path, label)

        if c == 100:
            if label is None:
                # 标签不存在
                pass
            else:
                if len(points) > 0:
                    cv2.fillPoly(label, [np.array(points)], (0, 0, 0))
                    masked = src2017.copy()
                    masked[label > 128] = masked[label > 128] * (1 - ALPHA) + np.array([(0, 0, 255)]) * ALPHA
                    cv2.imwrite(save_path, label)

    if c == 110:
        return 0
    elif c == 112:
        return 1
    elif c == 27:
        return -1


if __name__ == '__main__':
    # 按键说明
    # s: 115 应用多边形并保存
    # d: 100 消除多边形并保存
    # n: 110 切换到下一张(不保存)
    # p: 112 返回上一张(不保存)
    # esc: 27 退出程序
    # 鼠标右键撤销红点

    fn0 = []
    fn1 = []
    fn2 = []
    fn3 = []
    fn4 = []

    i = 0
    j = 0
    k = 0

    for i in range(0, IM_ROWS // ROI_SIZE + 1):
        for j in range(0, IM_COLS // ROI_SIZE):
            ss0 = "{}/2015/s2_{}_{}_{}.tif".format(DATA_DIR, i, j, ROI_SIZE)
            ss1 = "{}/2017/s2_{}_{}_{}.tif".format(DATA_DIR, i, j, ROI_SIZE)
            # ss2 = "{}/pred/s2_{}_{}_{}.tif".format(DATA_DIR, i, j, ROI_SIZE)
            ss3 = "{}/label_pred_818_remove_small/s2_{}_{}_{}.tif".format(DATA_DIR, i, j, ROI_SIZE)
            ss4 = "{}/pred_9_label/s2_{}_{}_{}.tif".format(DATA_DIR, i, j, ROI_SIZE)

            fn0.append(ss0)
            fn1.append(ss1)
            # fn2.append(ss2)
            fn3.append(ss3)
            fn4.append(ss4)

    sum = (i + 1) * (j + 1)

    while k < sum:
        src2015 = cv2.imread(fn0[k])
        src2017 = cv2.imread(fn1[k])
        # pred = cv2.imread(fn2[k], cv2.IMREAD_UNCHANGED)
        pred709 = cv2.imread(fn3[k], cv2.IMREAD_UNCHANGED)
        print("current image path:", fn4[k])
        status = label_img(src2015, src2017, pred709, fn4[k])

        if status == 0:
            k += 1
        elif status == 1:
            if k == 0:
                print("这已经是第一张了！")
            else:
                k -= 1
        elif status == -1:
            exit(0)
