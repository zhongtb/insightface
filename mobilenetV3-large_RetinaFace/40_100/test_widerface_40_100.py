from __future__ import print_function

import argparse
import sys
import os
import time
import numpy as np

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

import mxnet as mx
from mxnet import ndarray as nd
import cv2
from rcnn.logger import logger
#from rcnn.config import config, default, generate_config
#from rcnn.tools.test_rcnn import test_rcnn
#from rcnn.tools.test_rpn import test_rpn
from rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes, landmark_pred
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper
from rcnn.processing.bbox_transform import bbox_overlaps
from rcnn.dataset import retinaface
#from retinaface import RetinaFace
from retinaface_stride_8_16 import RetinaFace


def parse_args():
    parser = argparse.ArgumentParser(description='Test widerface by retinaface detector')
    # general
    parser.add_argument('--network', help='network name', default='net3', type=str)
    parser.add_argument('--dataset', help='dataset name', default='retinaface', type=str)
    parser.add_argument('--image-set', help='image_set name', default='val', type=str)
    parser.add_argument('--root-path', help='output data folder', default='./data', type=str)
    parser.add_argument('--dataset-path', help='dataset path', default='./data/retinaface', type=str)
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    # testing
    parser.add_argument('--prefix', help='model to test with', default='', type=str)
    parser.add_argument('--epoch', help='model to test with', default=0, type=int)
    parser.add_argument('--output', help='output folder', default='./wout', type=str)
    parser.add_argument('--nocrop', help='', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=0.02, type=float)
    parser.add_argument('--mode', help='test mode, 0 for fast, 1 for accurate', default=1, type=int)
    #parser.add_argument('--pyramid', help='enable pyramid test', action='store_true')
    #parser.add_argument('--bbox-vote', help='', action='store_true')
    parser.add_argument('--part', help='', default=0, type=int)
    parser.add_argument('--parts', help='', default=1, type=int)
    args = parser.parse_args()
    return args

detector = None
args = None
imgid = -1

def get_boxes(roi, pyramid):
  global imgid
  im = cv2.imread(roi['image'])
  do_flip = False
  if not pyramid:
    target_size = 1200
    max_size = 1600
    #do_flip = True
    target_size = 1504
    max_size = 2000
    target_size = 1600
    max_size = 2150
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
  else:
    do_flip = True
    #TEST_SCALES = [500, 800, 1200, 1600]
    TEST_SCALES = [500, 800, 1100, 1400, 1700]
    target_size = 800
    max_size = 1200
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    scales = [float(scale)/target_size*im_scale for scale in TEST_SCALES]
  scales = [1.0]
  boxes, landmarks = detector.detect(im, threshold=args.thresh, scales = scales, do_flip=do_flip)
  #print(boxes.shape, landmarks.shape)
  if imgid>=0 and imgid<100:
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in xrange(boxes.shape[0]):
      box = boxes[i]
      ibox = box[0:4].copy().astype(np.int)
      cv2.rectangle(im, (ibox[0], ibox[1]), (ibox[2], ibox[3]), (255, 0, 0), 2)
      #print('box', ibox)
      #if len(ibox)>5:
      #  for l in xrange(5):
      #    pp = (ibox[5+l*2], ibox[6+l*2])
      #    cv2.circle(im, (pp[0], pp[1]), 1, (0, 0, 255), 1)
      blur = box[5]
      k = "%.3f"%blur
      cv2.putText(im,k,(ibox[0]+2,ibox[1]+14), font, 0.6, (0,255,0), 2)
      #landmarks = box[6:21].reshape( (5,3) )
      if landmarks is not None:
        for l in xrange(5):
          color = (0,255,0)
          landmark = landmarks[i][l]
          pp = (int(landmark[0]), int(landmark[1]))
          if landmark[2]-0.5<0.0:
            color = (0,0,255)
          cv2.circle(im, (pp[0], pp[1]), 1, color, 2)
    filename = './testimages/%d.jpg'%imgid
    cv2.imwrite(filename, im)
    print(filename, 'wrote')
    imgid+=1

  return boxes


def test(args):
  print('test with', args)
  global detector
  output_folder = args.output
  if not os.path.exists(output_folder):
    os.mkdir(output_folder)
  detector = RetinaFace(args.prefix, args.epoch, args.gpu, network=args.network, nocrop=args.nocrop, vote=args.bbox_vote)
  print('args.dataset:',args.dataset)
  print('args.image_set:',args.image_set)
  print('args.root_path:',args.root_path)
  print('args.dataset_path:',args.dataset_path)
  imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
  print('imdb: ',imdb)
  roidb = imdb.gt_roidb()
  gt_overlaps = np.zeros(0)
  overall = [0.0, 0.0]
  gt_max = np.array( (0.0, 0.0) )
  num_pos = 0
  print('roidb size', len(roidb))
  pic_num = 3225
  box_all = 0
  box_last = 0
  for i in range(len(roidb)):
    if i%args.parts!=args.part:
      continue
    #if i%10==0:
    #  print('processing', i, file=sys.stderr)
    roi = roidb[i]
    boxes = get_boxes(roi, args.pyramid)
    if 'boxes' in roi:
      gt_boxes = roi['boxes'].copy()
      gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
      num_pos += gt_boxes.shape[0]

      overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
      #print(im_info, gt_boxes.shape, boxes.shape, overlaps.shape, file=sys.stderr)

      _gt_overlaps = np.zeros((gt_boxes.shape[0]))

      if boxes.shape[0]>0:
        _gt_overlaps = overlaps.max(axis=0)
        #print('max_overlaps', _gt_overlaps, file=sys.stderr)
        for j in range(len(_gt_overlaps)):
          if _gt_overlaps[j]>0.5:
            continue
          #print(j, 'failed', gt_boxes[j],  'max_overlap:', _gt_overlaps[j], file=sys.stderr)

        # append recorded IoU coverage level
        found = (_gt_overlaps > 0.5).sum()
        recall = found / float(gt_boxes.shape[0])
        #print('recall', _recall, gt_boxes.shape[0], boxes.shape[0], gt_areas, 'num:', i, file=sys.stderr)
        overall[0]+=found
        overall[1]+=gt_boxes.shape[0]
        #gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
        #_recall = (gt_overlaps >= threshold).sum() / float(num_pos)
        recall_all = float(overall[0])/overall[1]
        #print('recall_all', _recall, file=sys.stderr)
        print('[%d]'%i, 'recall', recall, (gt_boxes.shape[0], boxes.shape[0]), 'all:', recall_all, file=sys.stderr)
        print(pic_num-i,' : left')
    else:
      print('[%d]'%i, 'detect %d faces'%boxes.shape[0])


    _vec = roidb[i]['image'].split('/')
    out_dir = os.path.join(output_folder, _vec[-2])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_file = os.path.join(out_dir, _vec[-1].replace('jpg', 'txt'))
    with open(out_file, 'w') as f:
      name = '/'.join(roidb[i]['image'].split('/')[-2:])
      #print('name: ',name)
      f.write("%s\n"%(name))
      #f.write("%d\n"%(boxes.shape[0]))
      #print('boxes.shape[0]: ',boxes.shape[0])
      box_num = 0
      a = []
      #print('boxes: ',boxes)
      for b in range(boxes.shape[0]):
        box_all += 1
        box = boxes[b]
        w = box[2]-box[0]
        h = box[3]-box[1]
        if (w > 40 and w < 100) or (h > 40 and h < 100):
            box_last += 1
            box_num += 1
            a.append(box)
            #f.write("%d %d %d %d %g \n"%(box[0], box[1], box[2]-box[0], box[3]-box[1], box[4]))
      f.write("%d\n"%box_num)
      print('box_num: ',box_num)
      if len(a) > 0:
        for i in range(len(a)):
          box = a[i]#.astype(np.int)
            #box_num += 1
          f.write("%d %d %d %d %g \n"%(box[0], box[1], box[2]-box[0], box[3]-box[1], box[4]))
  print('box_all: ',box_all)
  print('box_last: ',box_last)

def main():
    global args
    args = parse_args()
    args.pyramid = False
    args.bbox_vote = False
    if args.mode==1:
      args.pyramid = True
      args.bbox_vote = True
    logger.info('Called with argument: %s' % args)
    test(args)

if __name__ == '__main__':
    main()

