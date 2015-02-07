# =================================================================================================
# @Author: Junbo Xin
# @Date: 2015/02/06
# @Description: stage wise regression
# =================================================================================================

from numpy import *
import regression


def rss_error(y_arr, y_hat_arr):
    return ((y_arr-y_hat_arr)**2).sum()


# initialize w as 1, for each features j(j=1,2,..,n), change a little, each time change a para,
# see if the error has been shrink or not
def stage_wise(x_arr, y_arr, eps=0.01, num_iter=100):
    # step 1: data normalization
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    y_mean = mean(y_mat, 0)
    x_mean = mean(x_mat, 0)
    x_var = var(x_mat, 0)
    y_mat = y_mat - y_mean
    x_mat = (x_mat - x_mean) / x_var

    # step 2: initialize w
    m, n = shape(x_mat)
    return_mat = zeros((num_iter, n))
    ws = zeros((n, 1))  # initialize all to 0
    ws_test = ws.copy()
    ws_max = ws.copy()

    # step 3: calculate w, each line is a vector of n elements, num_iter*n
    for i in range(num_iter):
        print ws.T
        lowest_error = inf
        # step 4: go through each features(n)
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign  # for each feature, add or subtract with a step eps
                y_test = x_mat * ws_test
                rss_err = rss_error(y_mat.A, y_test.A)
                # step 5: if current ws makes error smaller, then update ws_max
                if rss_err < lowest_error:
                    lowest_error = rss_err
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T  # each line of return_mat is a n vector
    return return_mat


def test_stage():
    x_arr, y_arr = regression.load_data_set('abalone.txt')
    stage_wise(x_arr, y_arr, 0.01, 200)
