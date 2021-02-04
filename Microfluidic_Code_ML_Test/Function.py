#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/22 01:36
# @Author  : Wei
# @Email   : sanqsunwei@gmail.com
# @File    : Function.py
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix


def smooth(data_in, box_pts):
    data = data_in.copy()
    box = np.ones(box_pts)/box_pts  # mean kernel (could change)
    data_smooth = np.convolve(data, box, mode='same')
    return data_smooth


# This function is for data loading
def fdc_data_loading(f_file_name, f_system_model, f_data_length):

    np_all_data = np.empty((0, f_data_length*3))
    np_all_label = np.empty((0, 1))

    # read data name
    # print(os.listdir('StudyData/'))

    data_name_list = os.listdir(f_file_name)

    data_name_list.sort()
    # print(f_file_name)
    for data_name in data_name_list:
        # print(data_name)
        label_buffer = data_name.split('_')

        # choose csv length
        if len(label_buffer) < 4:
            continue

        # choose csv channel (system model == control or test)
        system_model = 'C' + str(f_system_model)
        # if label_buffer[1] != 'C4':
        if label_buffer[1] != system_model:
            # print(label_buffer[1])
            continue

        # load label and data
        label_data = label_buffer[2]
        # print(label_data)
        raw_data = pd.read_csv(f_file_name + data_name, header=None)
        useful_data = raw_data.loc[1:, [2 + f_system_model, 6 + f_system_model, 10 + f_system_model]].to_numpy()
        # useful_data = raw_data.loc[1:, [2 + f_system_model]].to_numpy()

        useful_data = useful_data.astype(np.float64)
        # print(useful_data.dtype)

        # delete the data which too short
        if len(useful_data) <= f_data_length:
            continue
        data_to_1d = np.concatenate((useful_data[0:f_data_length, 0],
                                     useful_data[0:f_data_length, 1]/1000000, useful_data[0:f_data_length, 2]), axis=0)
        # data_to_1d = useful_data[0:f_data_length, 0]
        # combine all the data and the label in the file
        np_all_data = np.vstack((np_all_data, data_to_1d))
        np_all_label = np.vstack((np_all_label, label_data))

    return np_all_data, np_all_label


def svm_evaluation(f_np_data, f_np_label, f_folder_of_classifier):

    features = pd.DataFrame(f_np_data)
    labels = pd.DataFrame(f_np_label)
    labels.columns = ['label']
    f_data = pd.concat([labels, features], axis=1)
    f_data_groupby = f_data.groupby('label')
    f_data_groupby_list = list(f_data_groupby)

    f_length = len(f_np_data[0, :])
    np_data_training = np.empty((0, f_length + 1))
    np_data_testing = np.empty((0, f_length + 1))
    train_set = 0.5

    for i in range(len(f_data_groupby_list)):
        f_data_np = np.asarray(f_data_groupby_list[i][1])
        train_number = int(train_set * len(f_data_np))
        np_data_training = np.vstack((np_data_training, f_data_np[:train_number]))
        np_data_testing = np.vstack((np_data_testing, f_data_np[train_number:]))
    y_train, x_train = np_data_training[:, 0].reshape(-1, ), np_data_training[:, 1:]
    y_test, x_test = np_data_testing[:, 0].reshape(-1, ), np_data_testing[:, 1:]

    # print(x_train.shape)
    # print(y_test)

    # model
    print("Training the model")
    laser_classifier = svm.SVC(kernel='linear')
    laser_classifier.fit(x_train, y_train)
    _Y = laser_classifier.predict(x_test)

    print("Output the results")
    confu_matrix = pd.crosstab(pd.Series(y_test), pd.Series(_Y),
                               rownames=['True'], colnames=['Predicted'], margins=True)

    if not os.path.exists(f_folder_of_classifier):
        os.makedirs(f_folder_of_classifier)
    confu_matrix.to_csv(f_folder_of_classifier + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_confu.csv')
    pred_result = classification_report(_Y, y_test)
    system_log = open(f_folder_of_classifier + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_system_log.txt', "a")
    system_log.write(str(pred_result))
    result = np.sum(_Y == y_test) / (len(y_test))
    system_log.write(str(result))

    print("=========")
    print(confu_matrix)
    print("=========")
    print(pred_result)
    print("=========")
    print(result)
    print("=========")
    return result


def get_ten_set_of_time(f_np_data):
    ten_fold_set_list = []
    ten_fold_data_set_list = []
    n_length = len(f_np_data)
    # n_length = 8
    ten_fold = 10
    each_set_number = int(n_length / ten_fold)
    first_x_set_add = int(n_length % ten_fold)
    index_last_y = (each_set_number + 1) * first_x_set_add
    all_index_set = [i for i in range(n_length)]
    for i in range(ten_fold):
        if i < first_x_set_add:
            ten_fold_set_list.append(all_index_set[i * (each_set_number + 1):(i + 1) * (each_set_number + 1)])
            ten_fold_data_set_list.append(f_np_data[i * (each_set_number + 1):(i + 1) * (each_set_number + 1), :])
        else:
            ten_fold_set_list.append(all_index_set[index_last_y + (i - first_x_set_add) * each_set_number:
                                                   index_last_y + (i - first_x_set_add + 1) * each_set_number])
            ten_fold_data_set_list.append(f_np_data[index_last_y + (i - first_x_set_add) * each_set_number:
                                                    index_last_y + (i - first_x_set_add + 1) * each_set_number, :])
    if n_length < ten_fold:
        ten_fold_data_set_list = ten_fold_data_set_list[0:n_length]
        for i in range(ten_fold - n_length):
            ten_fold_set_list[n_length + i] = [n_length - 1]
            ten_fold_data_set_list.append((f_np_data[n_length - 1, :].reshape(1,-1)))

    # return ten_fold_set_list
    return ten_fold_data_set_list


def ten_fold_set_seg(f_np_data, f_np_label):

    # group each label
    features = pd.DataFrame(f_np_data)
    labels = pd.DataFrame(f_np_label)
    labels.columns = ['label']
    f_data = pd.concat([labels, features], axis=1)
    f_data_groupby = f_data.groupby('label')
    f_data_groupby_list = list(f_data_groupby)
    f_group_ten_set_list = []

    for i in range(len(f_data_groupby_list)):
        np_one_group = np.asarray(f_data_groupby_list[i][1])
        ten_set_data = get_ten_set_of_time(np_one_group)
        f_group_ten_set_list.append(ten_set_data)

    f_ten_fold_set = []
    f_length = len(f_np_data[0, :])
    for i in range(10):
        f_np_one_fold_set = np.empty((0, f_length + 1))
        for j in range(len(f_data_groupby_list)):
            f_np_one_fold_set = np.vstack((f_np_one_fold_set, f_group_ten_set_list[j][i]))
        f_ten_fold_set.append(f_np_one_fold_set)
    return f_ten_fold_set


def ten_fold_cross_validation(f_ten_fold_set_list, f_folder_of_classifier):
    accuracy_list = np.array([])
    confu_matrix = pd.DataFrame()
    for i in range(len(f_ten_fold_set_list)):
        y_test = f_ten_fold_set_list[i][:, 0].reshape(-1,)
        x_test = f_ten_fold_set_list[i][:, 1:]

        f_np_one_fold_set = np.empty((0, f_ten_fold_set_list[i].shape[1]))
        for j in range(len(f_ten_fold_set_list)):
            if j != i:
                f_np_one_fold_set = np.vstack((f_np_one_fold_set, f_ten_fold_set_list[j]))
        y_train = f_np_one_fold_set[:, 0]
        x_train = f_np_one_fold_set[:, 1:]

        confu_matrix_of_one_fold, accuracy_0f_one_fold = svm_evaluation_one_fold(y_train, x_train, y_test, x_test)

        for j in range(int(max(y_test))+1):
            columns_lost = str(j)
            if columns_lost not in list(confu_matrix_of_one_fold.columns):
                col_name = confu_matrix_of_one_fold.columns.tolist()
                col_name.insert(j, columns_lost)
                confu_matrix_of_one_fold = confu_matrix_of_one_fold.reindex(columns=col_name)
                confu_matrix_of_one_fold[columns_lost] = [0 for i in range(int(max(y_test)) + 2)]
        if i == 0:
            confu_matrix = confu_matrix.append(confu_matrix_of_one_fold)
        else:
            confu_matrix += confu_matrix_of_one_fold

        accuracy_list = np.append(accuracy_list, accuracy_0f_one_fold)

    mean_accuracy = sum(accuracy_list) / len(f_ten_fold_set_list)
    ordered_accuracy_list = np.sort(accuracy_list)
    print("=========")
    print("Mean Accuracy: {:.2f}%".format(100 * mean_accuracy))
    print("Rank5 Max Accuracy: {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%".
          format(100 * ordered_accuracy_list[-1], 100 * ordered_accuracy_list[-2],
                 100 * ordered_accuracy_list[-3], 100 * ordered_accuracy_list[-4],
                 100 * ordered_accuracy_list[-5]))
    print("Min Accuracy: {:.2f}%".format(100 * ordered_accuracy_list[0]))
    # print("=========")

    # confu_matrix = confu_matrix / len(f_ten_fold_set_list)
    # confu_matrix = confu_matrix.applymap(lambda x: '%.2f%%' % (x * 100))
    confu_matrix.to_csv(f_folder_of_classifier + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_confu.csv')
    print(confu_matrix)
    # print("=========")

    return 100 * mean_accuracy


def svm_evaluation_one_fold(y_train, x_train, y_test, x_test, ):

    f_classifier = svm.SVC(kernel='linear')
    f_classifier.fit(x_train, y_train)
    _Y = f_classifier.predict(x_test)

    f_confu_matrix = pd.crosstab(pd.Series(y_test), pd.Series(_Y),
                                 rownames=['True'], colnames=['Predicted'], margins=True)
    f_accuracy_0f_one_fold = np.sum(_Y == y_test) / (len(y_test))
    # pred_result = classification_report(_Y, y_test)
    # # print("=========")
    # print(pred_result)
    # print(f_confu_matrix)
    return f_confu_matrix, f_accuracy_0f_one_fold
