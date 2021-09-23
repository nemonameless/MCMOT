# -*- coding: utf-8 -*-
# *******************************************************************************
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# *******************************************************************************
"""
Authors: Li Jie, lijie47@baidu.com
Date:    2021/6/3 13:22
"""
from motmetrics.math_util import quiet_divide


__all__ = [
    'motp_overall',
    'mota_overall',
    'precision_overall',
    'recall_overall',
    'idp_overall',
    'idr_overall',
    'idf1_overall'
]


def motp_overall(summary_df, overall_dic):
    motp = quiet_divide((summary_df['motp'] * summary_df['num_detections']).sum(), overall_dic['num_detections'])
    return motp


def mota_overall(summary_df, overall_dic):
    del summary_df
    mota = 1. - quiet_divide((overall_dic['num_misses'] + overall_dic['num_switches'] + overall_dic['num_false_positives']),
                             overall_dic['num_objects'])
    return mota


def precision_overall(summary_df, overall_dic):
    del summary_df
    precision = quiet_divide(overall_dic['num_detections'], (overall_dic['num_false_positives'] + overall_dic['num_detections']))
    return precision


def recall_overall(summary_df, overall_dic):
    del summary_df
    recall = quiet_divide(overall_dic['num_detections'], overall_dic['num_objects'])
    return recall


def idp_overall(summary_df, overall_dic):
    del summary_df
    idp = quiet_divide(overall_dic['idtp'], (overall_dic['idtp'] + overall_dic['idfp']))
    return idp


def idr_overall(summary_df, overall_dic):
    del summary_df
    idr = quiet_divide(overall_dic['idtp'], (overall_dic['idtp'] + overall_dic['idfn']))
    return idr


def idf1_overall(summary_df, overall_dic):
    del summary_df
    idf1 = quiet_divide(2. * overall_dic['idtp'], (overall_dic['num_objects'] + overall_dic['num_predictions']))
    return idf1
