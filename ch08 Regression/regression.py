# =================================================================================================
# @Author: Junbo Xin
# @Date: 2015/02/04-05
# @Description: Simple Regression
# =================================================================================================

from numpy import *


def load_data_set(file_name):
    num = len(open(file_name).readline().split('\t'))-1
    data_mat = []
    label_mat = []
    fr = open(file_name).readlines()
    for line in fr:
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))

    return data_mat, label_mat


# for y = w^T * x,  w = (X^T*X)^-1 * X^T * y
def calc_weight(x_arr, y_arr):
    x_mat = mat(x_arr)      # m*n matrix
    y_mat = mat(y_arr).T    # m*1 matrix
    x = x_mat.T * x_mat     # n*n
    if linalg.det(x) == 0.0:
        print 'matrix is singular, cannot inverse'
        return
    ws = x.I * (x_mat.T * y_mat)  # (n*n) * (n*m * m*1) = n*1
    return ws


# giving the data set, plot the scatter figure and the estimate regression line y_hat = w * x
def plot_line():
    import matplotlib.pyplot as plt
    # obtain the weight: w
    x_arr, y_arr = load_data_set('ex0.txt')
    w = calc_weight(x_arr, y_arr)
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)

    # plot the scatter: the real value
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(array(x_arr)[:, 1], array(y_mat.T)[:, 0], c='red')
    # ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0], c='red')

    # plot the line, predict one of regression
    x_copy = array(x_mat.copy())
    x_copy.sort(0)
    y_hat = array(x_copy * w)
    ax.plot(x_copy[:, 1], y_hat)  # plot parameter should be array, not matrix

    # calculate the correlative coefficient
    y_hat = x_mat * w  # here y_hat is matrix, different from the previous array
    print corrcoef(y_hat.T, y_mat)

    plt.show()


#  w_hat = (X^T * W * X)* X^T * W * y
def local_weight_point_test(test_point, x_arr, y_arr, k=1.0):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    m, n = shape(x_mat)
    weights = mat(eye((m)))
    for i in range(m):
        diff_mat = test_point - x_mat[i, :]  # difference between test_point and current point( i in 0--m)
        weights[i, i] = exp(diff_mat*diff_mat.T/(-2.0*k**2))
    x = x_mat.T * (weights*x_mat)
    if linalg.det(x) == 0.0:
        print 'This matrix is singular, cannot do inverses'
        return
    ws = x.I * x_mat.T * weights * y_mat
    return test_point * ws


def local_weight_array_test(test_arr, x_arr, y_arr, k=1.0):
    m, n = shape(x_arr)
    y_hat = zeros(m)
    for i in range(m):
        y_hat[i] = local_weight_point_test(test_arr[i], x_arr, y_arr, k)
    return y_hat


def plot_line_lw():
    import matplotlib.pyplot as plt
    x_arr, y_arr = load_data_set('ex0.txt')
    y_hat = local_weight_array_test(x_arr, x_arr, y_arr, k=3)
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    sorted_index = x_mat[:, 1].argsort(0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x_copy = x_mat.copy()
    x_copy.sort(0)
    ax.scatter(array(x_mat[:, 1]), array(y_mat.T[:, 0]), c='red')
    ax.plot(array(x_copy[:, 1]), array(y_hat[sorted_index]))
    ax.plot()
    plt.show()


