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
    max_err = ops[0]
    min_num = ops[1]
    # if there's 1 feature, return
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)

    m, n = shape(data_set)
    err_total = err_type(data_set)
    best_err = inf
    best_index = 0
    best_value = 0
    for feat_index in range(n-1):
        for val in set(data_set[:, feat_index]):
            mat0, mat1 = bin_split_data(data_set, feat_index, val)
            if shape(mat0)[0] < min_num or shape(mat1)[0] < min_num:
                continue
            new_err = err_type(mat0) + err_type(mat1)
            if new_err < best_err:
                best_index = feat_index
                best_value = val
                best_err = new_err
    if err_total - best_err < max_err:
        return None, leaf_type(data_set)
    mat0, mat1 = bin_split_data(data_set, best_index, best_value)
    if (shape(mat0)[0] < min_num) or (shape(mat1)[0] < min_num):
        return None, leaf_type(data_set)
    return best_index, best_value


