# -*- coding: utf-8 -*-
# *******************************************************************************
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# *******************************************************************************
"""
Authors: Li Jie, lijie47@baidu.com
Date:    2021/6/2 22:53
"""
import sys
import pandas as pd
import motmetrics as mm
import lib.tracking_utils.mcmot_overall as mcmot_overall
from lib.tracking_utils.evaluation import EvaluatorMCMOT


metrics_list = [
    'num_frames',
    'num_matches',
    'num_switches',
    'num_transfer',
    'num_ascend',
    'num_migrate',
    'num_false_positives',
    'num_misses',
    'num_detections',
    'num_objects',
    'num_predictions',
    'num_unique_objects',
    'mostly_tracked',
    'partially_tracked',
    'mostly_lost',
    'num_fragmentations',
    'motp',
    'mota',
    'precision',
    'recall',
    'idfp',
    'idfn',
    'idtp',
    'idp',
    'idr',
    'idf1'
]

name_map = {
    'num_frames': 'num_frames',
    'num_matches': 'num_matches',
    'num_switches': 'IDs',
    'num_transfer': 'IDt',
    'num_ascend': 'IDa',
    'num_migrate': 'IDm',
    'num_false_positives': 'FP',
    'num_misses': 'FN',
    'num_detections': 'num_detections',
    'num_objects': 'num_objects',
    'num_predictions': 'num_predictions',
    'num_unique_objects': 'GT',
    'mostly_tracked': 'MT',
    'partially_tracked': 'partially_tracked',
    'mostly_lost': 'ML',
    'num_fragmentations': 'FM',
    'motp': 'MOTP',
    'mota': 'MOTA',
    'precision': 'Prcn',
    'recall': 'Rcll',
    'idfp': 'idfp',
    'idfn': 'idfn',
    'idtp': 'idtp',
    'idp': 'IDP',
    'idr': 'IDR',
    'idf1': 'IDF1'
}


def parse_accs_metrics(seq_acc, index_name, verbose=False):
    """
    从 motmetrics 类中解析多个MOTAccumulator的评估指标
    """
    mh = mm.metrics.create()
    summary = EvaluatorMCMOT.get_summary(seq_acc, index_name, metrics_list)
    summary.loc['OVERALL', 'motp'] = (summary['motp'] * summary['num_detections']).sum() / \
                                     summary.loc['OVERALL', 'num_detections']
    if verbose:
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=name_map
        )
        print(strsummary)

    return summary


def seqs_overall_metrics(summary_df, verbose=False):
    """
    计算多个序列中的 overall metrics
    """
    add_col = [
        'num_frames',
        'num_matches',
        'num_switches',
        'num_transfer',
        'num_ascend',
        'num_migrate',
        'num_false_positives',
        'num_misses',
        'num_detections',
        'num_objects',
        'num_predictions',
        'num_unique_objects',
        'mostly_tracked',
        'partially_tracked',
        'mostly_lost',
        'num_fragmentations',
        'idfp',
        'idfn',
        'idtp'
    ]
    calc_col = [
        'motp',
        'mota',
        'precision',
        'recall',
        'idp',
        'idr',
        'idf1'
    ]
    calc_df = summary_df.copy()

    overall_dic = {}
    for col in add_col:
        overall_dic[col] = calc_df[col].sum()

    for col in calc_col:
        overall_dic[col] = getattr(mcmot_overall, col + '_overall')(calc_df, overall_dic)

    overall_df = pd.DataFrame(overall_dic,index=['overall_calc'])
    calc_df = pd.concat([calc_df, overall_df])

    # 显示结果
    if verbose:
        mh = mm.metrics.create()
        str_calc_df = mm.io.render_summary(
            calc_df,
            formatters=mh.formatters,
            namemap=name_map
        )
        print(str_calc_df)

    return calc_df






