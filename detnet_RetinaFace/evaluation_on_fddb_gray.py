# -*- coding: utf-8 -*-

import argparse
import os
import cv2
import math
import sys
import numpy as np
sys.path.append('..')
# change the config as your need
#from config_farm import configuration_10_320_20L_5scales_v2 as cfg
import mxnet
from retinaface_val_gray import RetinaFace
#from predict import Predict

# set the proper symbol file and model file
symbol_file_path = 'model/mnet_net3/mnet_net3-symobl.json'
#symbol_file_path = '../symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
#model_file_path = '../saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1800000.params'
#model_file_path = 'model/mnet_net3/mnet_net3'
model_file_path = '../../model_test_gray/mnet_net3a_faster_mobilenet_gray/mnet_net3a_faster_mobilenet_gray'
#model_file_path = '../saved_model/configuration_10_320_20L_5scales_v2_2019-10-10-21-15-41/train_10_320_20L_5scales_v2_iter_2000000.params'
#detector = RetinaFace(prefix_path, epoch_num, args.gpuid, 'net3a',args.dense_anchor)
detector = RetinaFace(model_file_path, 80, 0, 'net3a',False)
'''
my_predictor = Predict(mxnet=mxnet,
                       symbol_file_path=symbol_file_path,
                       model_file_path=model_file_path,
                       ctx=mxnet.gpu(0),
                       receptive_field_list=cfg.param_receptive_field_list,
                       receptive_field_stride=cfg.param_receptive_field_stride,
                       bbox_small_list=cfg.param_bbox_small_list,
                       bbox_large_list=cfg.param_bbox_large_list,
                       receptive_field_center_start=cfg.param_receptive_field_center_start,
                       num_output_scales=cfg.param_num_output_scales)
'''

# set fddb root, the path should look like XXXX/originalPics
fddb_image_root = './FDDB/originalPics'
# set the list file path, the path should look like XXXX/FDDB-folds/annotatedList.txt
image_list_file = './FDDB/FDDB-folds/imList.txt'
result_file_name = './fddb_eva/fddb_' + os.path.basename(model_file_path).split('.')[0] + '_result.txt'
fin = open(image_list_file, 'r')
fout = open(result_file_name, 'w')
resize_scale = 1.0
scales = [resize_scale]
score_threshold = 0.11
NMS_threshold = 0.4
counter = 0
flip = False
for line in fin:
    line = line.strip('\n')

    im = cv2.imread(os.path.join(fddb_image_root, line + '.jpg'), cv2.IMREAD_GRAYSCALE)
    im = im[:,:np.newaxis]

    bboxes= detector.detect(im, score_threshold, scales, do_flip=flip)
    #bboxes= detector.detect(im, thresh, scales, do_flip=flip)
    #bboxes = my_predictor.predict(im, resize_scale=resize_scale, score_threshold=score_threshold, top_k=10000, NMS_threshold=NMS_threshold)

    # for bbox in bboxes:
    #     cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 1)
    # cv2.imshow('im', im)
    # cv2.waitKey()

    fout.write(line + '\n')
    if np.array(bboxes[0]).shape[0] == 0:
        print('No face')
        fout.write(str(0) + '\n')
        continue
    fout.write(str(np.array(bboxes[0]).shape[0]) + '\n')
    print('find',np.array(bboxes[0]).shape[0],'faces') 
    for bbox in bboxes[0]:
        fout.write('%d %d %d %d %.03f' % (
        math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1]),
        bbox[4] if bbox[4] <= 1 else 1) + '\n')
    counter += 1
    print('[%d] %s is processed.' % (counter, line))
fin.close()
fout.close()

