import argparse
import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface_val import RetinaFace

def generateTXT(args):
    thresh = 0.8
    scales = [1024, 1980]
    target_size = scales[0]
    max_size = scales[1]
    count = 1
    gpuid = args.gpuid
    flip = False
    epoch_num = args.epoch_num
    detector = RetinaFace(args.model_path, epoch_num, gpuid, 'net3a')

   #abs_dir = os.getcwd()
    #wider_val_dir = os.path.join(abs_dir, 'detect', 'WIDER', 'WIDER_val', 'images')
    f_dir = os.walk(args.data_path)  
    for path,dir_list,file_list in f_dir:  
        for dir_name in dir_list:
            #print('dirname: ', dir_name)
            imgs = []
            name_list = []
            scales = []
            scales_list = []
            txts_dir = os.path.join(args.save_path, dir_name)
            if not os.path.exists(txts_dir):
                os.makedirs(txts_dir)
            im_dir = os.path.join(path, dir_name)
            images_name= os.listdir(im_dir)
            for image_na in images_name:
                name_list.append(image_na[:-4])
                print('name: ', image_na[:-4])
                with open(os.path.join(txts_dir, image_na[:-4]+'.txt'), 'w') as f:
                    f.write(image_na[:-4]+'\n')
                    im = cv2.imread(os.path.join(im_dir, image_na))
                    im_shape = im.shape
                    im_size_min = np.min(im_shape[0:2])
                    im_size_max = np.max(im_shape[0:2])
                    im_scale = float(target_size) / float(im_size_min)
                    scales = [im_scale]
      #scales_list.append(scales)
                    faces, scores = detector.detect(im, thresh, scales, do_flip=flip)
                    if faces is not None:
                        print('find', faces.shape[0], 'faces')
                        f.write(str(faces.shape[0])+'\n')
                        for i in range(faces.shape[0]):
                            box = faces[i].astype(np.int)
                            f.write(str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]-box[0]) + ' ' + str(box[3]-box[1]) + ' ' + str(scores[i][0]) + '\n')
def parse_args():
    parser = argparse.ArgumentParser(description='RetinaFace VAL TXT')
    # general
    parser.add_argument('--model_path', help='model_path', default='model/retina', type=str)
    parser.add_argument('--data_path', help='wider_val_dir', default='detect/WIDER/WIDER_val/images', type=str)
    parser.add_argument('--gpuid', help='gpuid', default=0, type=int)
    parser.add_argument('--save_path', help='save_path', default='result_txt/wider_val', type=str)
    parser.add_argument('--epoch_num', help='epoch_num', default='0', type=int)
    args = parser.parse_args()
    return args

def main():
  args = parse_args()
  #logger.info('Called with argument: %s' % args)
  generateTXT(args)

if __name__ == '__main__':
    main()


