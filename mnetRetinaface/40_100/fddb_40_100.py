import argparse
import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface_val import RetinaFace
import time

def fddb_pre_txt(args):
    prefix_path = args.model_path  #模型路径及名称
    epoch_num = args.epoch_num
    data_path = args.data_path  #测试数据集路径
    save_dir = args.save_path   #生成的TXT保存路径文件夹

    thresh = 0.8#0.5#0.8
    scales = [1024, 1980]
    target_size = scales[0]
    max_size = scales[1]

#count = 1
    t1 = time.time()
    gpuid = 2
    flip = False
    detector = RetinaFace(prefix_path, epoch_num, args.gpuid, args.network,args.dense_anchor)

    abs_dir = os.getcwd()
    count =1
    img_num = 2845
    #print('共{}张图片'.format(img_mun))i
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, 'results.txt'), 'w')  as f:
        for i in range(10):
            t2 = time.time()
            print('all:{}'.format(t2 - t1))
            print('Processing...... ', i+1)
            if i == 9:
                txt_dir = os.path.join(data_path,'FDDB-folds', 'FDDB-fold-10.txt')
            else:
                txt_dir = os.path.join(data_path, 'FDDB-folds', 'FDDB-fold-0'+str(i+1)+'.txt')
  #print('txt_dir: ',txt_dir)
            with open(txt_dir,'r') as f1:
                lines = f1.readlines()
                for line in lines:
                    t1 = time.time()
                    
      #print('#######*****', i)
                    line = line.strip('\n\t')
                    img_path = os.path.join(abs_dir, 'FDDB/originalPics',line+'.jpg')
      #print('img_path:',img_path)
                    f.write(line+'\n')
                    im = cv2.imread(img_path)
                    im_shape = im.shape
                    im_size_min = np.min(im_shape[0:2])
                    im_size_max = np.max(im_shape[0:2])
                    im_scale = float(target_size) / float(im_size_min)
                    scales = [im_scale]
                    scales = [1.0] 
                    faces, scores = detector.detect(im, thresh, scales, do_flip=flip)
                    t3 = time.time()
                    print('face_all: ',faces)
                    if faces.shape[0] == 0:
                    #if faces is None:
                        f.write(str(0)+'\n')
                    else:
                        print('find', faces.shape[0], 'faces')
                        box_num = 0
                        a = []
                        for i in range(faces.shape[0]):
                            box = faces[i].astype(np.int)
                            w = box[2]-box[0]
                            h = box[3]-box[1]
                            if (w > 40 and w < 100) or (h > 40 and h < 100):
                                box_num += 1
                                a.append(box)
                        #print('box_num: ',box_num)
                        f.write(str(box_num)+'\n')
                        #f.write(str(faces.shape[0])+'\n')
                        for i in range(len(a)):
                            box = a[i].astype(np.int)  #faces[i].astype(np.int)
                            f.write(str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]-box[0]) + ' ' + str(box[3]-box[1]) + ' ' + str(scores[i][0]) + '\n')
                    count +=1
                    print('cout{} , last{}'.format(count, img_num- count))

def parse_args():
    parser = argparse.ArgumentParser(description='RetinaFace VAL TXT')
    # general
    parser.add_argument('--model_path', help='model_path', default='model/retina', type=str)
    parser.add_argument('--data_path', help='FDDB_val_dir', default='FDDB', type=str)
    parser.add_argument('--network', help='FDDB_network', default='net3a', type=str)
    parser.add_argument('--gpuid', help='gpuid', default=0, type=int)
    parser.add_argument('--save_path', help='save_path', default='FDDB_pre_txt/wider_val', type=str)
    parser.add_argument('--epoch_num', help='epoch_num', default='0', type=int)
    parser.add_argument('--dense_anchor', help='dense_mode', default=False, type=bool)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    #print('参数个数为:', len(sys.argv), '个参数。')
    #p#rint('参数列表:', str(sys.argv))
    #print('脚本名为：', sys.argv[0])
    #for i in range(1, len(sys.argv)):
        #print('参数 %s 为：%s' % (i, sys.argv[i]))
    fddb_pre_txt(args)

if __name__ == "__main__":
    main()
