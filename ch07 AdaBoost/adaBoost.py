# =================================================================================================
# @Author: Junbo Xin
# @Date: 2015/01/29
# @Description: Adaboost algorithm
# =================================================================================================
from numpy import *


def load_simple_data():
    data_mat = mat([[1.0, 2.1],
                   [2.0, 1.1],
                   [1.3, 1.0],
                   [1.0, 1.1],
                   [2.0, 1.0]])
    data_label = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, data_label


# giving data set, return the single classifier
def stump_classify(data_mat, dimen, threshold, inequa):
    m, n = data_mat.shape
    ret_arr = ones((m, 1))
    if inequa == 'less than':
        ret_arr[data_mat[:, dimen] > threshold] = -1.0
    else:
        ret_arr[data_mat[:, dimen] > threshold] = -1.0
    return ret_arr



