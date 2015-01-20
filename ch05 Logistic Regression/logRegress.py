
# ====================================================================
# @Author: Junbo Xin
# @Date: 2015/01/20
# @Description:   Logistic Regressions
# @More details:  http://blog.csdn.net/dongtingzhizi/article/details/15962797
# ====================================================================

from numpy import *
import matplotlib.pyplot as plt


#  100 examples: [x1, x2, y]
def load_data_set():
    data_mat = []
    label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        str = line.strip().split()
        # data_mat:[x0=1.0, x1, x2], 100 * 3
        data_mat.append([1.0, float(str[0]), float(str[1])])
        label_mat.append(int(str[2]))  # label_mat: 1 * 100
    return data_mat, label_mat   # return format: list


def sigmoid(input_x):
    return 1.0/(1+exp(-input_x))   # exp is the Numpy function, not in python


# Input: data set and label are lists   Output: the parameters: weights(vector)
def grad_ascent(data_input, label_input):
    # step 1: convert list to Numpy matrix
    data_mat = mat(data_input)     # 100*3
    label_mat = mat(label_input).transpose()
    rows, cols = shape(data_mat)
    alpha = 0.001
    max_iter = 500
    weights = ones((cols, 1))   # 3*1 vector

    for i in range(max_iter):
        # theta_j := theta_j - alpha * sum [(sigmoid(x(i)-y(i))*x_j(i)]
        h = sigmoid(data_mat * weights)  # 100*1 vector (100*3) * (3*1)
        error = (label_mat - h)
        weights += alpha * data_mat.transpose() * error  # (3*1) + (3*100)*(100*1) = 3*1
    return weights   # weights is Numpy array


# plot the data set and the LR classifier. Notice: the data set should be array, not matrix
def plot_fig(weights):
    data_mat, label_mat = load_data_set()
    data_arr = array(data_mat)
    w = weights  # .getA()   # turn to array
    n = shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-w[0]-w[1]*x)/w[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# no need to change the input to Numpy matrix or array
def sto_grad_ascent(data_input, label_input):
    rows, cols = shape(data_input)
    alpha = 0.01
    weights = ones(cols)
    for i in range(rows):
        h = sigmoid(data_input[i]*weights)  # real value: (1*3) * (3*1)
        error = label_input[i] - h
        weights += alpha * error * data_input[i]
    return weights


def sto_grad_ascent_improve(data_input, label_input, iter_num=150):
    rows, cols = shape(data_input)
    weights = ones(cols)
    for i in range(iter_num):
        data_index = range(rows)
        for j in range(rows):
            alpha = 4/(1.0+i+j) + 0.0001
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_input[rand_index]*weights))
            error = label_input[rand_index] - h
            weights += alpha * error * error * data_input[rand_index]
            del(data_index[rand_index])
    return weights





