"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed
import time


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']  #legend_name 10 1, pr_cruve2 1000
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        #lines = lines[1:]
        lines = lines[2:]

    boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')[0:5]], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):   #返回字典
    if os.path.exists(pred_dir):
        print(pred_dir)
    events = os.listdir(pred_dir)
    #print('events: ', events)
    boxes = dict()
    pbar = tqdm.tqdm(events)  #进度条库

    for event in pbar:

        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        #print('event_dir::',event_dir)  #/prediction/21--Festival
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))  #return img_file.split('/')[-1], boxes
            current_event[imgname.rstrip('.jpg')] = _boxes
        #print('event: ', event)
        #print('cureent_event::',current_event)
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])
    #print('image_eval_gt: ',_gt)
    #print('_pred.shape[0]:',_pred.shape[0])
    #print('_gt_size_befor:', _gt.shape)
    #idx = []
    #for i in range(_gt.shape[0]):
    #    if _gt[i,2] > 40 or _gt[i,2] > 100 or _gt[i,3] < 40 or _gt[i,3] > 100:
    #        idx.append(i)
    #_gt = np.delete(_gt, idx)
    #print('_gt_size_after:', _gt.shape)
    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
        #print('recall_list: ',recall_list)
        r_keep_index = np.where(recall_list == 1)[0]
        #print('r_keep_index:', h,' : ',len(r_keep_index))
        pred_recall[h] = len(r_keep_index)
    #print('recall_list: ',recall_list)
    #print('r_keep_index: ',r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    #print('mrec: ',mrec)
    #print('mpre: ',mpre)

    # compute the precision envelope
    #print('mpre.size: ',mpre.size)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path ,save_result,model_name,iou_thresh=0.5):
    #print('pred00: ', pred)
    pred = get_preds(pred)
    #print('pred: ', pred)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]#[hard_gt_list, medium_gt_list, easy_gt_list]#[easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:   #61
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            #print('img_list: ',img_list[:2])
            #print('event_name: ',event_name)
            pred_list = pred[event_name]
            #print('pred_list : ',pred_list )
            sub_gt_list = gt_list[i][0]
            #print('sub_gt_list: ',sub_gt_list[:2])
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]
            #print('pbar_gt: ',gt_bbx_list)

            for j in range(len(img_list)):   #115
                pred_info = pred_list[str(img_list[j][0][0])]  #一张图每个人脸的五个数据
               
                keep_index = sub_gt_list[j][0]
                #print(' keep_index_first: ', keep_index)
                gt_boxes = gt_bbx_list[j][0].astype('float')
                if len(gt_boxes) == 0:
                    continue
                #print(' gt_boxes: ', gt_boxes.shape[0])
                new_gt_boxes = []
                del_num = 0
                list1 = []
                list2 = []
                for i in range(gt_boxes.shape[0]):
                    w = gt_boxes[i][2]
                    h = gt_boxes[i][3]
                    if (w > 40 and w < 100) or (h > 40 and h < 100):
                        new_gt_boxes.append(gt_boxes[i])
                        del_num +=1
                        list1 = []
                        for idx in range(len(keep_index)):
                            arr = keep_index[idx]
                            #print('arr ',idx,' : ',arr)
                            if int(i+1) == int(arr[0]):
                                list1.append(arr[0]-(i+1-del_num))
                                list2.append(list1)
                                #print('lise1: ',list1)
                                #print('iii1: ',i+1)
                                #print('del_num: ',i+1-del_num)
                                break
                keep_index = np.array(list2)
               
                gt_boxes = new_gt_boxes.copy()
                gt_boxes = np.array(gt_boxes)
                #print(' new_gt_boxes: ', gt_boxes.shape[0])
                #keep_index = sub_gt_list[j][0]
                #print(' keep_index_type: ', type(keep_index))
                #print(' keep_index_new: ', keep_index)
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])  #每张#一共几个人脸
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1  #
                #print('gt_boxes: ',gt_boxes)
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    print("Medium Val AP: {}".format(aps[1]))
    print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")

    with open(os.path.join(save_result,'test_widerface_40_100.txt'), 'a') as f:
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(str(localtime)+'\n')
        f.write(model_name+'\n')
        f.write('Easy   Val AP: ' + str(aps[0]) + '\n')
        f.write('Medium   Val AP: ' + str(aps[1]) + '\n')
        f.write('Hard   Val AP: ' + str(aps[2]) + '\n')
        f.write('###########################################'+'\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', '--prediction', default='../result_txt/wider_val')
    parser.add_argument('--gt', '--groundtruth', default='../eval_tools/ground_truth')
    parser.add_argument('--save_result', '--save_result', default='../')
    parser.add_argument('--model_name', '--model_name', default='retina-0000')

    args = parser.parse_args()
    evaluation(args.pred, args.gt, args.save_result, args.model_name)












