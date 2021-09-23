# -*- coding: utf-8 -*-
# *******************************************************************************
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# *******************************************************************************
"""
Authors: Li Jie, lijie47@baidu.com
Date:    2021/9/22 13:56
"""
import os
import numpy as np
import copy
import motmetrics as mm
mm.lap.default_solver = 'lap'

from tracking_utils.io import read_results, unzip_objs, unzip_objs_cls


class Evaluator(object):

    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == 'mot'

        # gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        gt_filename = os.path.join(self.data_root, 'annotations', '{}.txt'.format(self.seq_name))
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True, multi_class=True, union=True)
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]
        #match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
        #match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
        #match_ious = iou_distance[match_is, match_js]

        #match_js = np.asarray(match_js, dtype=int)
        #match_js = match_js[np.logical_not(np.isnan(match_ious))]
        #keep[match_js] = False
        #trk_tlwhs = trk_tlwhs[keep]
        #trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_frame_cls(self, frame_id, trk_tlwhs, trk_ids, trk_cls, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)
        trk_cls = np.copy(trk_cls)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids, gt_cls = unzip_objs_cls(gt_objs)[:3]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # 将不同类别对象之间的距离设置为nan
        gt_cls = np.copy(gt_cls)
        gt_cls_len = len(gt_cls)
        trk_cls_len = len(trk_cls)
        # 当 gt 和 trk 又一个数目为0时，iou_distance 维度为(0,0)
        if gt_cls_len != 0 and trk_cls_len != 0:
            gt_cls = gt_cls.reshape(gt_cls_len, 1)
            gt_cls = np.repeat(gt_cls, trk_cls_len, axis=1)
            trk_cls = trk_cls.reshape(1, trk_cls_len)
            trk_cls = np.repeat(trk_cls, gt_cls_len, axis=0)
            iou_distance = np.where(gt_cls == trk_cls, iou_distance, np.nan)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()

        result_frame_dict = read_results(filename, self.data_type, is_gt=False, multi_class=True, union=True)
        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            # trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            # self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)
            trk_tlwhs, trk_ids, trk_cls = unzip_objs_cls(trk_objs)[:3]
            self.eval_frame_cls(frame_id, trk_tlwhs, trk_ids, trk_cls, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()


class EvaluatorMCMOT(object):

    def __init__(self, class_num, gt_filename, data_type):
        """
        init
        """
        self.class_num = class_num
        self.gt_filename = gt_filename
        self.data_type = data_type

        self.reset_accumulator()
        self.class_accs = []
        self.case_accs = []

    def reset_accumulator(self):
        """
        reset_accumulator
        """
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame_dict(self, trk_objs, gt_objs, rtn_events=False, union=False):
        if union:
            trk_tlwhs, trk_ids, trk_cls = unzip_objs_cls(trk_objs)[:3]
            gt_tlwhs, gt_ids, gt_cls = unzip_objs_cls(gt_objs)[:3]

            # get distance matrix
            iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

            # 将不同类别对象之间的距离设置为nan
            gt_cls_len = len(gt_cls)
            trk_cls_len = len(trk_cls)
            # 当 gt 和 trk 又一个数目为0时，iou_distance 维度为(0,0)
            if gt_cls_len != 0 and trk_cls_len != 0:
                gt_cls = gt_cls.reshape(gt_cls_len, 1)
                gt_cls = np.repeat(gt_cls, trk_cls_len, axis=1)
                trk_cls = trk_cls.reshape(1, trk_cls_len)
                trk_cls = np.repeat(trk_cls, gt_cls_len, axis=0)
                iou_distance = np.where(gt_cls == trk_cls, iou_distance, np.nan)

        else:
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

            # get distance matrix
            iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'mot_events'):
            events = self.acc.mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename):
        """
        按类别分别进行评估
        """
        gt_frame_dict = read_results(self.gt_filename, self.data_type, is_gt=True, multi_class=True)
        result_frame_dict = read_results(filename, self.data_type, is_gt=False, multi_class=True)

        # 按类别统计每个类别的跟踪评估指标
        for cid in range(self.class_num):
            self.reset_accumulator()

            cls_result_frame_dict = result_frame_dict.setdefault(cid, dict())
            cls_gt_frame_dict = gt_frame_dict.setdefault(cid, dict())

            # 仅评估有标注帧
            frames = sorted(list(set(cls_gt_frame_dict.keys())))

            for frame_id in frames:
                trk_objs = cls_result_frame_dict.get(frame_id, [])
                gt_objs = cls_gt_frame_dict.get(frame_id, [])
                self.eval_frame_dict(trk_objs, gt_objs, rtn_events=False)

            self.class_accs.append(self.acc)

        return self.class_accs

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()
