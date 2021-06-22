import sys
import os
import time
import math
import torch
import numpy as np
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size

from tool import utils


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def get_region_boxes(boxes_and_confs):
    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)

    return [boxes, confs]


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def do_three_detect(model1, img1, img2, img3, conf_thresh, nms_thresh, use_cuda=0):
    model1.eval()

    if type(img1) == np.ndarray and len(img1.shape) == 3:  # cv2 image
        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img1) == np.ndarray and len(img1.shape) == 4:
        img1 = torch.from_numpy(img1.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)
    if use_cuda:
        img1 = img1.cuda()
    img1 = torch.autograd.Variable(img1)

    if type(img2) == np.ndarray and len(img2.shape) == 3:  # cv2 image
        img2 = torch.from_numpy(img2.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img2) == np.ndarray and len(img2.shape) == 4:
        img2 = torch.from_numpy(img2.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img2 = img2.cuda()
    img2 = torch.autograd.Variable(img2)

    if type(img3) == np.ndarray and len(img3.shape) == 3:  # cv2 image
        img3 = torch.from_numpy(img3.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img3) == np.ndarray and len(img3.shape) == 4:
        img3 = torch.from_numpy(img3.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img3 = img3.cuda()
    img3 = torch.autograd.Variable(img3)
    output1 = model1(img1)
    output2 = model1(img2)
    output3 = model1(img3)
    return utils.post_processing(img1, conf_thresh, nms_thresh, output1), utils.post_processing(img2, conf_thresh,
                                                                                                nms_thresh,
                                                                                                output2), utils.post_processing(
        img3, conf_thresh, nms_thresh, output3)


def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=0):
    model.eval()
    t0 = time.time()

    if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    t1 = time.time()
    output = model(img)
    t2 = time.time()

    # print('-----------------------------------')
    # print('         Preprocessed : %f' % (t1 - t0))
    # print('      Model Inference : %f' % (t2 - t1))
    # print('-----------------------------------')

    return utils.post_processing(img, conf_thresh, nms_thresh, output)
