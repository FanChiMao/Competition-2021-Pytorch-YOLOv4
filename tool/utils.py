import math
import os
import time
import numpy as np
from torch import nn
import math
import os
import time
from shapely import geometry
import numpy as np
from torch import nn


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    # print('iou box1:', box1)
    # print('iou box2:', box2)

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def majority_vote_boxes_set(img, boxes1, boxes2, boxes3):
    from itertools import zip_longest
    import numpy as np
    output_boxes_first = []
    output_boxes_second = []
    output_boxes_third = []
    output_boxes_fourth = []
    img = np.copy(img)
    # width = img.shape[1]
    # height = img.shape[0]

    zipboxes = list(zip_longest(boxes1, boxes2, boxes3, fillvalue=[0, 0, 0, 0, 0, 0, 0]))  # 缺少的去做補洞
    # print('  Model1: %d' % len(boxes1))
    # print('  Model2: %d' % len(boxes2))
    # print('  Model3: %d' % len(boxes3))
    print('Voting pass.....', end='')
    t0 = time.time()
    for num_1 in range(len(zipboxes)):  # 最多個偵測點的model
        # example = [[x1,y1],[x2,y2],[x3,y3]]  x1 =example[0][0]
        Flag_first = True
        Flag_second = True

        X1x1_main = zipboxes[num_1][0][0]
        X1y1_main = zipboxes[num_1][0][1]
        X1x2_main = zipboxes[num_1][0][2]
        X1y2_main = zipboxes[num_1][0][3]
        #
        X2x1_main = zipboxes[num_1][1][0]
        X2y1_main = zipboxes[num_1][1][1]
        X2x2_main = zipboxes[num_1][1][2]
        X2y2_main = zipboxes[num_1][1][3]

        for num_2 in range(len(zipboxes)):
            X2x1 = zipboxes[num_2][1][0]
            X2y1 = zipboxes[num_2][1][1]
            X2x2 = zipboxes[num_2][1][2]
            X2y2 = zipboxes[num_2][1][3]

            X2_center_x, X2_center_y = (X2x1 + X2x2) * 0.5, (X2y1 + X2y2) * 0.5

            X3x1 = zipboxes[num_2][2][0]
            X3y1 = zipboxes[num_2][2][1]
            X3x2 = zipboxes[num_2][2][2]
            X3y2 = zipboxes[num_2][2][3]

            X3_center_x, X3_center_y = (X3x1 + X3x2) * 0.5, (X3y1 + X3y2) * 0.5

            if Flag_first:  # 1 是否 與2或3重疊
                if X1x1_main < X2_center_x < X1x2_main and X1y1_main < X2_center_y < X1y2_main or \
                        X1x1_main < X3_center_x < X1x2_main and X1y1_main < X3_center_y < X1y2_main:
                    output_boxes_first.append(
                        [X1x1_main, X1y1_main, X1x2_main, X1y2_main, zipboxes[num_1][0][4], zipboxes[num_1][0][5],
                         zipboxes[num_1][0][6]])
                    Flag_first = False
            if Flag_second:  # 2 是否 與 3 重疊
                if X2x1_main < X3_center_x < X2x2_main and X2y1_main < X3_center_y < X2y2_main:
                    output_boxes_second.append(
                        [X2x1_main, X2y1_main, X2x2_main, X2y2_main, zipboxes[num_1][1][4], zipboxes[num_1][1][5],
                         zipboxes[num_1][1][6]])
                    Flag_second = False

            if Flag_first is False and Flag_second is False:
                break

    # if [0, 0, 0, 0, 0, 0, 0] in output_boxes_first: #移除補洞用的fillValue
    # output_boxes_first.remove([0, 0, 0, 0, 0, 0, 0])

    # if [0, 0, 0, 0, 0, 0, 0] in output_boxes_second: #移除補洞用的fillValue
    # output_boxes_second.remove([0, 0, 0, 0, 0, 0, 0])
    output_boxes_first = list(filter(([0, 0, 0, 0, 0, 0, 0]).__ne__, output_boxes_first))
    output_boxes_second = list(filter(([0, 0, 0, 0, 0, 0, 0]).__ne__, output_boxes_second))

    for x in range(len(output_boxes_second)):
        x1 = output_boxes_second[x][0]
        y1 = output_boxes_second[x][1]
        x2 = output_boxes_second[x][2]
        y2 = output_boxes_second[x][3]
        x_center, y_center = (x1 + x2) * 0.5, (y1 + y2) * 0.5

        for y in range(len(output_boxes_first)):
            xx1 = output_boxes_first[y][0]
            yy1 = output_boxes_first[y][1]
            xx2 = output_boxes_first[y][2]
            yy2 = output_boxes_first[y][3]

            if xx1 < x_center < xx2 and yy1 < y_center < yy2:
                output_boxes_third.append(
                    [x1, y1, x2, y2, output_boxes_second[x][4], output_boxes_second[x][5],
                     output_boxes_second[x][6]])
                break
    t1 = time.time()
    print('%.2f seconds' % (t1 - t0))
    output_boxes_fourth = [x for x in output_boxes_second if x not in output_boxes_third]
    # output_boxes_fourth =output_boxes_second - output_boxes_third
    # output_boxes_fourth = output_boxes_second.differance(output_boxes_third)

    output_boxes_first.extend(output_boxes_fourth)
    return list(output_boxes_first)


def plot_boxes_and_create_csv(img, boxes, img_name=None, img_path=None, csv_path=None, SetBoundary=0, SetArea=0,
                              mask=None, class_names=None, color=None):
    import csv
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    import cv2
    # print(GAP(img))
    img = np.copy(img)

    # mask = np.copy(mask)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]

    total_confidence = 0
    count = 0
    with open(csv_path + "/" + img_name + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            boundary_height = y2 - y1
            boundary_width = x2 - x1
            area = boundary_height * boundary_width
            center_x = round((x1 + x2) * 0.5)
            center_y = round((y1 + y2) * 0.5)

            if color:
                rgb = color
            else:
                rgb = (0, 0, 255)

            cls_conf = box[5]
            cls_id = box[6]
            # print('%s: %f' % (class_names[cls_id], cls_conf))
            # classes = len(class_names)
            # offset = cls_id * 123457 % classes
            # red = get_color(2, offset, classes)
            # green = get_color(1, offset, classes)
            # blue = get_color(0, offset, classes)
            # 去除boundary problem和角落小框
            k = SetBoundary
            if boundary_width <= k * boundary_height or boundary_height <= k * boundary_width or area < SetArea:
                pass
            elif mask is not None:  # 這張圖有mask的話
                ############################################
                # numpy 和 opencv 的座標相反， numpy:(H,W) / cv2:(W,H)
                ############################################
                if ((mask[center_y, center_x] == 0) ).any():
                    # if (mask[y1:y2, x1:x2] == 0).any():
                    pass
                else:
                    # cv2.rectangle(影像, 頂點作標, 對像頂點作標, 顏色, 寬度)
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 3)  # (255, 0, 0)
                    # 框框外有文字類別敘述
                    img = cv2.putText(img, str(round(cls_conf, 4)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 2)
                    count += 1
                    w.writerow([center_x, center_y])
                    total_confidence += cls_conf
            else:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 3)  # (255, 0, 0)
                img = cv2.putText(img, str(round(cls_conf, 4)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 2)
                count += 1
                w.writerow([center_x, center_y])
                total_confidence += cls_conf
        try:
            average_confidence = total_confidence / count
        except:
            average_confidence = 0
            print('no detected point!')

        if img_name:
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            cv2.imwrite(img_path + "/" + img_name + '.JPG', img)

    f.close()
    print('Average confidence score = %.4f' % average_confidence)
    print('Total prediction numbers: %d' % count)
    return count


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None, path=None, SetBoundary=0, area_C=0,
                   mask=None):
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]

    total_confidence = 0
    count = 0
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        boundary_height = y2 - y1
        boundary_width = x2 - x1
        area = boundary_height * boundary_width
        center_x = round((x1 + x2) * 0.5)
        center_y = round((y1 + y2) * 0.5)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            # print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
        # 去除boundary problem和角落小框
        k = SetBoundary
        if boundary_width <= k * boundary_height or boundary_height <= k * boundary_width or area < area_C:
            pass
        elif mask is not None:  # 這張圖有mask的話
            ############################################
            # numpy 和 opencv 的座標相反， numpy:(H,W) / cv2:(W,H)
            ############################################
            if (mask[center_y, center_x] == 0).any():
                # if (mask[y1:y2, x1:x2] == 0).any():
                pass
            else:
                # cv2.rectangle(影像, 頂點作標, 對像頂點作標, 顏色, 寬度)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 3)  # (255, 0, 0)
                # 框框外有文字類別敘述
                img = cv2.putText(img, str(round(cls_conf, 4)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 2)
                count += 1
                total_confidence += cls_conf
        else:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 3)  # (255, 0, 0)
            img = cv2.putText(img, str(round(cls_conf, 4)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 2)
            count += 1
            total_confidence += cls_conf
    try:
        average_confidence = total_confidence / count
    except:
        average_confidence = 0

    print('Average confidence score = %.4f' % average_confidence)
    if savename:
        if not os.path.exists(path):
            os.mkdir(path)
        cv2.imwrite(path + "/" + savename, img)
        # print("save plot results to %s" % savename)
        # cv2.imwrite(savename, img)

    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def post_processing(img, conf_thresh, nms_thresh, output):
    box_array = output[0]
    confs = output[1]

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]
    box_array = box_array[:, :, 0]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)
    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k],
                         ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)
    # print('-----------------------------------')
    # print('       max and argmax : %f' % (t2 - t1))
    # print('                  nms : %f' % (t3 - t2))
    # print('Post processing total : %f' % (t3 - t1))
    # print('-----------------------------------')

    return bboxes_batch


def ray_tracing(x, y, poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside
