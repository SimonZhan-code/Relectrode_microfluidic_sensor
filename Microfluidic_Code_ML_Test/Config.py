#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 01:56
# @Author  : Wei
# @Email   : sanqsunwei@gmail.com
# @File    : Config.py
# @Software: PyCharm

# data_file_name = '../StudyData/Data_FDC_0714/'
# data_file_name = '../StudyData/Data_FDC/'
# data_file_name = 'StudyData/Data_LCD_0826/'
data_file_name = 'StudyData/Data_LCD_0828/'
control_channel_number = 1
# test_channel_number = 3
# test2_channel_number = 4
data_length = 60
smooth_box = 10
folder_of_classifier = 'StudyData/Model_SVM/'
samplerate = 12
seg_window = samplerate * 4 # 4 seconds
lowpass_filter_order = 1
cutoff = 1