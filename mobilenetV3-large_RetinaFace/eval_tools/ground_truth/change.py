from scipy.io import loadmat
import os
import tqdm
#import pickle
import argparse
import numpy as np

def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list

if __name__ == '__main__':
    gt_dir = './'
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_dir)
    for i in range(2):
        print('******************************************')
        print('facelist:',i,' : ',facebox_list[i])
        print('event_list',i,' : ',event_list[i])
        print('file_list',i,': ',file_list[i])
        print('hard_gt_list,',i,' : ',hard_gt_list[i])
        print('medium_gt_list',i,' : ',medium_gt_list[i])
        print('easy_gt_list',i,' : ',[i])
