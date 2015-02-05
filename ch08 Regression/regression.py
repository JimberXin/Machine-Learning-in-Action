# =================================================================================================
# @Author: Junbo Xin
# @Date: 2015/02/04-05
# @Description: Simple Regression
# =================================================================================================

from numpy import *


# giving a training file, return the list format: data_mat, label_mat
def load_data_set(file_name):
    num = len(open(file_name).readline().split('\t'))-1
    data_arr = []
    label_arr = []
    fr = open(file_name).readlines()
    for line in fr:
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num):
            line_arr.append(float(cur_line[i]))
        data_arr.append(line_arr)
        label_arr.append(float(cur_line[-1]))

    return data_arr, label_arr


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
    # step 1: obtain the weight: w
    x_arr, y_arr = load_data_set('ex0.txt')
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    w = calc_weight(x_arr, y_arr)

    # step 2: plot the scatter: the real value
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(array(x_arr)[:, 1], array(y_mat.T)[:, 0], c='red')
    # ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0], c='red')

    # step 3: plot the line, predict one of regression
    x_copy = array(x_mat.copy())
    x_copy.sort(0)
    y_hat = array(x_copy * w)
    ax.plot(x_copy[:, 1], y_hat)  # plot parameter should be array, not matrix

    # step 4: calculate the correlative coefficient
    y_hat = x_mat * w  # here y_hat is matrix, different from the previous array
    print corrcoef(y_hat.T, y_mat)

    plt.show()


# giving the single test point, the training data(x_arr and y_arr), and k of Gauss kernel, return y predict
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
    ws = x.I * x_mat.T * weights * y_mat   # w_hat = (X^T * W * X)* X^T * W * y
    return test_point * ws


# giving a test_arr(m*n) and training data(x_arr, y_arr), return a predictive vector y_hat(m*1)
def local_weight_array_test(test_arr, x_arr, y_arr, k=1.0):
    m, n = shape(x_arr)
    y_hat = zeros(m)
    for i in range(m):
        y_hat[i] = local_weight_point_test(test_arr[i], x_arr, y_arr, k)
    return y_hat


#
def plot_line_lw():
    import matplotlib.pyplot as plt
    # step 1: data preprocess
    x_arr, y_arr = load_data_set('ex0.txt')
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)
    sorted_index = x_mat[:, 1].argsort(0)  # sort the x_mat of the 2nd col, for plot figure
    x_copy = x_mat.copy()
    x_copy.sort(0)

    # step 2: plot the scatter: the real value
    fig = plt.figure()
    ax1 = fig.add_subplot(311)  # subplot(a,b,c): rows, cols, current num. all < 10, ignore ','
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.scatter(array(x_mat[:, 1]), array(y_mat.T[:, 0]), c='yellow')
    ax2.scatter(array(x_mat[:, 1]), array(y_mat.T[:, 0]), c='yellow')
    ax3.scatter(array(x_mat[:, 1]), array(y_mat.T[:, 0]), c='yellow')

    # step 3: plot different types of regressions with different k
    y_hat1 = local_weight_array_test(x_arr, x_arr, y_arr, k=3)
    ax1.plot(array(x_copy[:, 1]), array(y_hat1[sorted_index]), c='red', linewidth=1.2)
    plt.sca(ax1)
    plt.xlim(0, 1.0)
    plt.ylim(3.0, 5.0)
    plt.title('k=3')

    y_hat2 = local_weight_array_test(x_arr, x_arr, y_arr, k=0.05)
    ax2.plot(array(x_copy[:, 1]), array(y_hat2[sorted_index]), c='red', linewidth=1.2)
    plt.sca(ax2)
    plt.xlim(0, 1.0)
    plt.ylim(3.0, 5.0)
    plt.title('k=0.05')

    y_hat3 = local_weight_array_test(x_arr, x_arr, y_arr, k=0.003)
    ax3.plot(array(x_copy[:, 1]), array(y_hat3[sorted_index]), c='red', linewidth=1.2)
    plt.sca(ax3)
    plt.xlim(0, 1.0)
    plt.ylim(3.0, 5.0)
    plt.title('k=0.003')
    plt.show()


