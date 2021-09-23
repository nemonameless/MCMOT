# -*- coding: utf-8 -*-
# *******************************************************************************
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# *******************************************************************************
"""
Authors: Li Jie, lijie47@baidu.com
Date:    2021/9/22 13:58
"""
import os
from typing import Dict
import numpy as np

from lib.tracking_utils.log import logger


def write_results(filename, results_dict: Dict, data_type: str):
    if not filename:
        return
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if data_type in ('mot', 'mcmot', 'lab'):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, frame_data in results_dict.items():
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0)
                f.write(line)
    logger.info('Save results to {}'.format(filename))


def read_results(filename, data_type: str, is_gt=False, is_ignore=False, multi_class=False, union=False):
    if data_type in ('mot', 'lab'):
        if multi_class:
            if union:
                # 将所有类别联合进行评估(不同类别间的 track id 不能重复)
                read_fun = read_mcmot_results_union
            else:
                # 按类别将结果分开进行评估
                read_fun = read_mcmot_results
        else:
            read_fun = read_mot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)


"""
labels={'ped', ...			% 1
'person_on_vhcl', ...	% 2
'car', ...				% 3
'bicycle', ...			% 4
'mbike', ...			% 5
'non_mot_vhcl', ...		% 6
'static_person', ...	% 7
'distractor', ...		% 8
'occluder', ...			% 9
'occluder_on_grnd', ...		%10
'occluder_full', ...		% 11
'reflection', ...		% 12
'crowd' ...			% 13
};
"""


def read_mot_results(filename, is_gt, is_ignore):
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())

                if is_gt:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    score = 1
                elif is_ignore:
                    if 'MOT16-' in filename or 'MOT17-' in filename:
                        label = int(float(linelist[7]))
                        vis_ratio = float(linelist[8])
                        if label not in ignore_labels and vis_ratio >= 0:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def read_mcmot_results_union(filename, is_gt, is_ignore):
    results_dict = dict()
    if os.path.isfile(filename):
        all_result = np.loadtxt(filename, delimiter=',')
        if all_result.shape[0] == 0 or all_result.shape[1] < 7:
            return results_dict
        if is_ignore:
            return results_dict
        if is_gt:
            # 线下测试使用
            all_result = all_result[all_result[:, 7] != 0]
            all_result[:, 7] = all_result[:, 7] - 1

        if all_result.shape[0] == 0:
            return results_dict

        class_unique = np.unique(all_result[:, 7])

        last_max_id = 0
        result_cls_list = []
        for cls in class_unique:
            result_cls_split = all_result[all_result[:, 7] == cls]
            result_cls_split[:, 1] = result_cls_split[:, 1] + last_max_id
            # 保证每个类别的 track id 各不相同
            last_max_id = max(np.unique(result_cls_split[:, 1])) + 1
            result_cls_list.append(result_cls_split)

        results_con = np.concatenate(result_cls_list)

        for line in range(len(results_con)):
            linelist = results_con[line]
            fid = int(linelist[0])
            if fid < 1:
                continue
            results_dict.setdefault(fid, list())

            if is_gt:
                score = 1
            else:
                score = float(linelist[6])

            tlwh = tuple(map(float, linelist[2:6]))
            target_id = int(linelist[1])
            cls = int(linelist[7])

            results_dict[fid].append((tlwh, target_id, cls, score))

        return results_dict


def read_mcmot_results(filename, is_gt, is_ignore):
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                cid = int(linelist[7])
                if is_gt:
                    score = 1
                    # 线下测试用 线上删除
                    cid -= 1
                else:
                    score = float(linelist[6])

                cls_result_dict = results_dict.setdefault(cid, dict())
                cls_result_dict.setdefault(fid, list())

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                cls_result_dict[fid].append((tlwh, target_id, score))

    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores


def unzip_objs_cls(objs):
    """
    多类别目标跟踪数据解析
    """
    if len(objs) > 0:
        tlwhs, ids, cls, scores = zip(*objs)
    else:
        tlwhs, ids, cls, scores = [], [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    ids = np.array(ids)
    cls = np.array(cls)
    scores = np.array(scores)

    return tlwhs, ids, cls, scores


def gen_bad_labels(badcase_type, frame_objs):
    """
    生成每帧所对应的bad_labels 检测框以及其对应的信息
    """
    bad_labels = []
    for j in range(len(badcase_type)):
        if badcase_type[j] == -1:
            continue
        tlwh = frame_objs[j][0]
        label_name = frame_objs[j][2]
        loc_x1 = tlwh[0]
        loc_y1 = tlwh[1]
        loc_x2 = loc_x1 + tlwh[2]
        loc_y2 = loc_y1 + tlwh[3]
        bad_labels.append({
            'x1': int(loc_x1),
            'y1': int(loc_y1),
            'x2': int(loc_x2),
            'y2': int(loc_y2),
            'pred_label_name': label_name,
            'id': int(frame_objs[j][1]),
            'bad_case_type': int(badcase_type[j]),
            'score': float(frame_objs[j][3])
        })

    return bad_labels
