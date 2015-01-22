
# =============================================================================
# @Author: Junbo Xin
# @Date: 2015/01/20
# @Description:   Logistic Regressions
# @More details:  http://blog.csdn.net/dongtingzhizi/article/details/15962797
# ============================================================================

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


# improved gradient ascent algorithm to
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
            weights += alpha * error * data_input[rand_index]
            del(data_index[rand_index])
    return weights


def classify(input_vec, weights):
    prob = sigmoid(sum(input_vec*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# function to test the colic illness of horse
def coli_test():
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    training_set = []
    training_label = []

    # training of the training file
    for line in fr_train.readlines():
        cur_line = line.strip().split('\t')
        line_arr = []
        # each line has 21 features
        for i in range(21):
            line_arr.append(float(cur_line[i]))
        training_set.append(line_arr)
        training_label.append(float(cur_line[21]))  # the 22th col is label
    # get the training weights from improved algorithm
    training_weights = sto_grad_ascent_improve(array(training_set), training_label, 500)

    # test of the file
    error_count = 0
    num_test = 0.0
    for line in fr_test.readlines():
        num_test += 1.0
        cur_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(cur_line[i]))
        if int(classify(array(line_arr), training_weights)) != \
           int(cur_line[21]):
            error_count += 1
    error_rate = (float(error_count)/num_test)
    print 'the error rate of this test is: %f' % error_rate
    return error_rate


# calc 10 times average error rates
def multi_test():
    num_test = 10
    error = 0.0
    for i in range(num_test):
        error += coli_test()
    print 'after %d iterations, the average error rate is:%f' % \
    (num_test, error/float(num_test))



