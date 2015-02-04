# ================================================================================================================
# @Author: Junbo Xin
# @Date: 2015/02/03
# @Description: Applications of ROC and horse disease predict using Adaboost algorithm
# ================================================================================================================

import adaBoost
from numpy import *


def load_data_file(file_name):
    num = len(open(file_name).readline().split('\t'))
    data_mat = []
    label_mat = []
    fr = open(file_name).readlines()
    for line in fr:
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num-1):
            line_arr.append(float(cur_line[i]))  # first num-1 cols are features
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))    # the last col is the label, different from LR, here label is -1 or 1
    return data_mat, label_mat


# load the txt file and test the adaboost algorithm
def test_file():
    # training process
    data_arr, label_arr = load_data_file('horseColicTraining2.txt')
    weak_classifier = adaBoost.ada_boost_train(data_arr, label_arr, 10)

    # test process
    test_arr, test_label = load_data_file('horseColicTest2.txt')
    predict_y = adaBoost.ada_classifier(test_arr, weak_classifier)
    m, n = shape(test_arr)
    error_arr = mat(ones((m, 1)))
    error_count = error_arr[predict_y != mat(test_label).T].sum()
    test_error_rate = float(error_count) / m
    print " \nTest error rate is: %f " % test_error_rate


# ROC figure:   x label is FP/(FP+TN);   y label is TP/(TP+FN). All are Predictive positive
def plot_roc(predict_strength, label_arr):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    y_sum = 0.0
    # calculate the positive sample in the predict set
    num_pos = sum(array(label_arr) == 1.0)
    y_step = 1 / float(num_pos)
    x_step = 1 / float(len(label_arr) - num_pos)
    sorted_index = predict_strength.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_index.tolist()[0]:
        # from (1.0, 1.0) to (0.0, 0.0)
        if label_arr[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        ax.plot([cur[0], cur[0]-del_x], [cur[1], cur[1]-del_y], c='b')
        cur = (cur[0]-del_x, cur[1]-del_y)
    ax.plot([0, 1], [0, 1], 'r--')   # plot the random line: y = x
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for adaboost horse colic detection sysmte')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print 'the area under the curve is: ', y_sum*x_step


def test_roc():
    data_arr, label_arr = load_data_file('horseColicTraining2.txt')
    weak_classifier, class_weight = adaBoost.ada_boost_train(data_arr, label_arr, 10)
    plot_roc(class_weight.T, label_arr)
