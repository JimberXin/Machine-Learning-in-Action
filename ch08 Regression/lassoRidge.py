# =================================================================================================
# @Author: Junbo Xin
# @Date: 2015/02/06
# @Description: lasso regression
# =================================================================================================

from numpy import *


def stage_wise(x_arr, y_arr, eps=0.0, num_iter=100):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mat = regularize(x_mat)

    m, n = shape(x_mat)
    return_mat = zeros((num_iter, n))
    ws = zeros((n, 1))
    for i in range(num_iter):
        print ws.T
        lowest_error = inf
        for j in range(n):
