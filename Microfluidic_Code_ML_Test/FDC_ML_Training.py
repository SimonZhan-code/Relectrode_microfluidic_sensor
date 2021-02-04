#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 01:51
# @Author  : Wei
# @Email   : sanqsunwei@gmail.com
# @File    : FDC_ML_Training.py
# @Software: PyCharm

import Config
import Function
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt



def feature_extract_np(f_list):
    # (uH, data, MHz)
    list_features = []
    for sub_list in f_list:
        # print(sub_list.shape)
        x = np.zeros((sub_list.shape[0], 5))
        x[:, 0] = np.amax(sub_list, axis=1)
        x[:, 1] = np.mean(sub_list, axis=1)
        x[:, 2] = np.std(sub_list, axis=1)
        x[:, 3] = np.median(sub_list, axis=1)
        for i in range(sub_list.shape[0]):
            x[i, 4] = 0
            for j in sub_list[i]:
                if j >= x[i, 1]:
                    x[i, 4] = x[i, 4] + 1
        list_features.append(x)
    np_features = np.concatenate((list_features[0], list_features[1], list_features[2]), axis=1)
    return np_features


def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':

    '''
    Part 1. Load Data
    1.1 choose file name (data file, model file )
    1.2 choose the length of segmentation window 
        * there's three usefully column data (uH, data, MHz), So the length of the load data = segmentation window * 3
    1.3 choose the channel which used for the data collection
    1.4 load data
    1.5 output the log of the system
    '''

    # the rawdata file name
    data_file_name = Config.data_file_name

    # the file to save the ML and ML results
    folder_of_classifier = Config.folder_of_classifier

    # the length of segmentation window
    data_length = Config.data_length

    # which channel we will load
    system_model = Config.control_channel_number
    # system_model = Config.test_channel_number

    # load the data and label
    np_data, np_label = Function.fdc_data_loading(data_file_name, system_model, data_length)

    print("Part 1.1 Loaded the raw data shape: (%s, %s)"% (np_data.shape))
    print("Part 1.2 Loaded the label data len: %s" % len(np_label))

    set_of_label = set()
    for item in np_label:
        for i in item:
            set_of_label.add(i)
    # print(len(set_of_label))
    class_number = len(set_of_label)
    print("Part 1.3 The number of type in the label is : %s" % class_number)

    # # plot the data to check (It's for  )
    # color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'g', 'b', 'b', ]
    # for i in range(type_number):
    #     plt.plot(np_data[i * type_number:(i + 1) * type_number, :].transpose(), color_list[i])
    # plt.show()

    '''
    Part 2. Denoising Data
    2.1 choose the sample rate of the data 
    2.2 choose the order of the lowpass filter
    2.3 choose the cutoff Hz 
    2.4 denoising data and saved to a list
    '''

    # samplerate = Config.samplerate
    # lowpass_filter_order = Config.lowpass_filter_order
    # cutoff = Config.cutoff
    # seg_window = Config.seg_window
    # data_length = Config.data_length
    #
    # list_denoising = []
    # for j in range(3): # (uH, data, MHz)
    #     np_data_denoising= np_data[:, j * data_length : j * data_length + seg_window].copy()
    #     for i in range(np_data.shape[0]):
    #         np_data_denoising[i, :] = butter_lowpass_filter(
    #             np_data_denoising[i, :], cutoff, samplerate, lowpass_filter_order)
    #     # color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'g', 'b', 'b', ]
    #     # for i in range(type_number):
    #     #     plt.plot(np_data_denoising[i * type_number:(i + 1) * type_number, :].transpose(), color_list[i])
    #     # plt.show()
    #     list_denoising.append(np_data_denoising[:,int(np_data_denoising.shape[1]/2):])
    #
    # print("Part 2.1 The shape of denoising list is : %s %s %s"
    #       % (len(list_denoising), len(list_denoising[0]), len(list_denoising[0][1])))
    #
    # np_denoising = np.concatenate((list_denoising[0], list_denoising[1], list_denoising[2]), axis=1)
    # print("Part 2.2 The shape of denoising np is : %s %s" % (np_denoising.shape))

    '''
    Part 3. Extract Features
    
    '''

    # np_features = feature_extract_np(list_denoising)
    # print("Part 3.1 The shape of feature np is : %s %s" % (np_features.shape))

    # if len(np_data) == len(np_label) and len(np_data) > 1:
    #     result = Function.svm_evaluation(np_features[:, 0:15], np_label, folder_of_classifier)
    #     print(result)

    # if len(np_data) == len(np_label) and len(np_data) > 1:
    #     result = Function.svm_evaluation(np_denoising, np_label, folder_of_classifier)
    #     print(result)

    # if len(np_data) == len(np_label) and len(np_data) > 1:
    #     result = Function.svm_evaluation(np_data[:, 0:60], np_label, folder_of_classifier)
    #     print(result)
    # #
    # if len(np_data) == len(np_label) and len(np_data) > 1:
    #     result = Function.svm_evaluation(np_data, np_label, folder_of_classifier)
    #     print(result)

    '''
    Part 4. Ten Fold Corss Vilidation
    '''
    np_input_data = np_data
    # np_input_data = np_data[:, 0:60]
    # np_input_data = np_features
    # np_input_data = np_denoising

    ten_fold_set_list = Function.ten_fold_set_seg(np_input_data, np_label)
    mean_accuracy = Function.ten_fold_cross_validation(ten_fold_set_list, folder_of_classifier)

