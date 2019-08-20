from __future__ import print_function
import sys
import mxnet as mx
import numpy as np

from rcnn.config import config


def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss', 'rpn_label', 'rpn_bbox_weight']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label



class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self, pred_idx=-1, label_idx=-1,name='RPNAcc'):
        super(RPNAccMetric, self).__init__(name)
        self.pred, self.label = get_rpn_names()
        #self.name = 'RPNAcc'
        self.name = [name, name+'_BG', name+'_FG']
        self.pred_idx = pred_idx
        self.label_idx = label_idx
        self.STAT = [0, 0, 0]

    def reset(self):
        """Clear the internal statistics to initial state."""
        if isinstance(self.name, str):
          self.num_inst = 0
          self.sum_metric = 0.0
        else:
          #print('reset to ',len(self.name), self.name, file=sys.stderr)
          self.num_inst = [0] * len(self.name)
          self.sum_metric = [0.0] * len(self.name)


    def get(self):
        if isinstance(self.name, str):
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(len(self.name))]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

    def update(self, labels, preds):
        if self.pred_idx>=0 and self.label_idx>=0:
          pred = preds[self.pred_idx]
          label = preds[self.label_idx]
        else:
          pred = preds[self.pred.index('rpn_cls_prob')]
          label = labels[self.label.index('rpn_label')]
          #label = preds[self.pred.index('rpn_label')]

        num_images = pred.shape[0]
        #print(pred.shape, label.shape, file=sys.stderr)
        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        #pred_label = pred_label.reshape((pred_label.shape[0], -1))
        pred_label = pred_label.reshape(-1,)
        # label (b, p)
        label = label.asnumpy().astype('int32').reshape(-1,)
        #print(pred_label.shape, label.shape)

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        #print('in_metric', pred_label.shape, label.shape, len(keep_inds), file=sys.stderr)
        #print(keep_inds, file=sys.stderr)
        _pred_label = pred_label[keep_inds]
        _label = label[keep_inds]
        #print('in_metric2', pred_label.shape, label.shape, len(keep_inds), file=sys.stderr)
        if isinstance(self.name, str):
          self.sum_metric += np.sum(_pred_label.flat == _label.flat)
          self.num_inst += len(_pred_label.flat)
        else:
          self.sum_metric[0] += np.sum(_pred_label.flat == _label.flat)
          self.num_inst[0] += len(_pred_label.flat)

          keep_inds = np.where(label == 0)[0]
          _pred_label = pred_label[keep_inds]
          _label = label[keep_inds]
          self.sum_metric[1] += np.sum(_pred_label.flat == _label.flat)
          self.num_inst[1] += len(_pred_label.flat)

          keep_inds = np.where(label == 1)[0]
          _pred_label = pred_label[keep_inds]
          _label = label[keep_inds]
          a = np.sum(_pred_label.flat == _label.flat)
          b = len(_pred_label.flat)
          self.sum_metric[2] += a
          self.num_inst[2] += b

          #self.STAT[0]+=a
          #self.STAT[1]+=b
          #self.STAT[2]+=num_images
          #if self.STAT[2]%400==0:
          #  print('FG_ACC', self.pred_idx, self.STAT[2], self.STAT[0], self.STAT[1], float(self.STAT[0])/self.STAT[1], file=sys.stderr)
          #  self.STAT = [0,0,0]


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, pred_idx=-1, label_idx=-1):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()
        self.pred_idx = pred_idx
        self.label_idx = label_idx

    def update(self, labels, preds):
        if self.pred_idx>=0 and self.label_idx>=0:
          pred = preds[self.pred_idx]
          label = preds[self.label_idx]
        else:
          pred = preds[self.pred.index('rpn_cls_prob')]
          label = labels[self.label.index('rpn_label')]
          #label = preds[self.pred.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]
        #print('in_metric log', label.shape, cls.shape, file=sys.stderr)

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, loss_idx=-1, weight_idx=-1, name='RPNL1Loss'):
        super(RPNL1LossMetric, self).__init__(name)
        self.pred, self.label = get_rpn_names()
        self.loss_idx = loss_idx
        self.weight_idx = weight_idx
        self.name = name

    def update(self, labels, preds):
        if self.loss_idx>=0 and self.weight_idx>=0:
          bbox_loss = preds[self.loss_idx].asnumpy()
          bbox_weight = preds[self.weight_idx].asnumpy()
        else:
          bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()
          bbox_weight = labels[self.label.index('rpn_bbox_weight')].asnumpy()
          #bbox_weight = preds[self.pred.index('rpn_bbox_weight')].asnumpy()

        #print('in_metric', self.name, bbox_weight.shape, bbox_loss.shape)

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / (bbox_weight.shape[1]/config.NUM_ANCHORS)
        #print('in_metric log', bbox_loss.shape, num_inst, file=sys.stderr)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

#landmark五个点准确度评估
class LandmarkDIFFMetric(mx.metric.EvalMetric):
    def __init__(self, tg_idx=-1, weight_idx = -1, diff_idx=-1, anchor_idx=-1, name='LandmarkDIFF'):
        super(LandmarkDIFFMetric, self).__init__(name)
        self.pred, self.label = get_rpn_names()
        self.tg_idx = tg_idx
        self.weight_idx = weight_idx
        self.diff_idx = diff_idx
        #self.anchor_idx = anchor_idx
        self.name = name

    def update(self, labels, preds):
        if self.tg_idx>=0 and self.weight_idx>=0:
          landmark_weight = preds[self.weight_idx].asnumpy()
          landmark_target = labels[self.tg_idx].asnumpy()
          landmark_diff = preds[self.diff_idx].asnumpy()
          #anchor = preds[self.anchor_idx].asnumpy()
        else:
          landmark_weight = labels[self.label.index('rpn_landmark_weight')].asnumpy()
          landmark_target = labels[self.label.index('face_landmark_target')].asnumpy()
          landmark_diff = preds[self.pred.index('rpn_landmark_diff')].asnumpy()
          #anchor = preds[self.label.index('anchors')].asnumpy()
          #bbox_weight = preds[self.pred.index('rpn_bbox_weight')].asnumpy()

        landmark_pred_len = 10
        if config.USE_OCCLUSION:
          landmark_pred_len = 15
        landmark_target = landmark_target.astype(np.float, copy=False).transpose(0,2,1).reshape(-1,landmark_pred_len)
        landmark_weight = landmark_weight.astype(np.float, copy=False).transpose(0,2,1).reshape(-1,landmark_pred_len)
        landmark_diff = landmark_diff.astype(np.float, copy=False).transpose(0,2,1).reshape(-1,landmark_pred_len)
        #anchor = anchor.astype(np.float, copy=False).transpose(0,2,1).reshape(-1,4)
        index = np.where(landmark_weight[:,0]==1)[0]  #landmark_weight[:,1]等等价，
        '''   
        widths = anchor[index, 2] - anchor[index, 0] + 1.0
        heights = anchor[index, 3] - anchor[index, 1] + 1.0
        ctr_x = anchor[index, 0] + 0.5 * (widths - 1.0)
        ctr_y = anchor[index, 1] + 0.5 * (heights - 1.0)
        '''
        gt_dx = abs(landmark_target[index,0] - landmark_target[index,2])#*widths  #两眼间距离
        #print('gt_dx:', min(gt_dx))
        #print('sum_gt_dx:', sum(gt_dx))
        diffs = []
        if landmark_pred_len == 10:
          for i in range(0,landmark_pred_len,2):   #range(10)
              diff = np.sum(np.sqrt(np.square(landmark_diff[index,i]) + np.square(landmark_diff[index,i+1])))/np.sum((gt_dx + (2e-10)))
              diffs.append(diff)
        elif landmark_pred_len == 15:
          for i in range(0,landmark_pred_len,3):   #range(10)
              diff = np.sqrt(np.square(landmark_diff[index,i]) + np.square(landmark_diff[index,i+1]))/(gt_dx+ (2e-10))
              diffs.append(diff)
        num_inst = landmark_pred_len //2
        #print('in_metric log', bbox_loss.shape, num_inst, file=sys.stderr)

        self.sum_metric += np.sum(diffs)
        self.num_inst += num_inst
