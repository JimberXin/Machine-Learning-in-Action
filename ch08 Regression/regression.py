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
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    x = x_mat.T * x_mat
    if linalg.det(x) == 0.0:
        print 'matrix is singular, cannot inverse'
        return
    ws = x.I * (x_mat.T * y_mat)
    return ws


def plot_line():
    import matplotlib.pyplot as plt
    # obtain the weight: w
    x_arr, y_arr = load_data_set('ex0.txt')
    w = calc_weight(x_arr, y_arr)
    x_mat = mat(x_arr)
    y_mat = mat(y_arr)


    # plot the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_mat[:, 1].flatten().A[0], y_mat.T[:, 0].flatten().A[0])
    x_copy = x_mat.copy()
    x_copy.sort(0)
    y_hat = x_copy * w
    ax.plot(x_copy[:, 1], y_hat)
    plt.show()