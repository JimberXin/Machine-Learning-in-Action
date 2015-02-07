# =============================================================================
# Author: Junbo Xin
# Date: 2015/02/07
# Description:  Regression Tree
# =============================================================================

from numpy import *


def load_data_set(file_name):
    data_mat = []
    fr = open(file_name).readlines()
    for line in fr:
        cur_line = line.strip().split('\t')
        float_line = map(float, cur_line)
        data_mat.append(float_line)
    return data_mat


def bin_split_data(data_set, feature, value):
    # for a 2-d array A, nonzero(A) returns array(m,n), m is the rows, n is the cols
    mat0 = data_set[nonzero(data_set[:, feature] > value)[0], :][0]
    mat1 = data_set[nonzero(data_set[:, feature] <= value)[0], :][0]
    return mat0, mat1


def reg_leaf(data_set):
    return mean(data_set[:, -1])


def reg_error(data_set):
    return var(data_set[:, -1]) * shape(data_set)[0]


def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_error, ops=(1, 4)):
    



