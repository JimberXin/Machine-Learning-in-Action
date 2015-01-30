# =================================================================================================
# @Author: Junbo Xin
# @Date: 2015/01/29-30
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


# get the single decision tree classifier: a m * 1 vector, i.e, [1, -1, 1, 1, 1,...,1]
# boundary:  condition 1):  if x[dimen]_i < threshold, y_i = 1, else y_i = -1
#            condition 2):  if x[dimen]_i > threshold, y_i = 1, else y_i = -1
def stump_classify(data_mat, dimen, threshold, inequal):
    ret_arr = ones((shape(data_mat)[0], 1))  # the whole vector initializes with 1
    if inequal == 'less than':
        ret_arr[data_mat[:, dimen] <= threshold] = -1.0
    else:
        ret_arr[data_mat[:, dimen] > threshold] = -1.0
    return ret_arr


def build_stump(data_arr, label_arr, weight_arr):
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).T
    m, n = shape(data_mat)
    num_step = 10.0
    best_stump = {}   # decision tree info
    label_predict = mat(zeros((m, 1)))   # a m*1 vector, 1 or -1
    min_error = inf
    #  step 1: go through each dimension of data set, i.e, each col
    for i in range(n):
        min_val = data_mat[:, i].min()
        max_val = data_mat[:, i].max()
        step_size = (max_val - min_val) / num_step

        # step 2: go through each threshold of this dimension
        for j in range(-1, int(num_step)+1):

            # step 3: go through both equality: <  and  >
            for inequal in ['less than', 'greater than']:
                threshold = min_val + float(j)*step_size
                predict_label = stump_classify(data_mat, i, threshold, inequal)
                error_arr = mat(ones((m, 1)))
                error_arr[predict_label == label_mat] = 0
                weight_error = weight_arr.T * error_arr
                print 'dimension: %d, threshold: %2.f, inequal: %s, weight error: %f' %\
                      (i, threshold, inequal, weight_error)

                # update min_error and best_stump if needed
                if weight_error < min_error:
                    min_error = weight_error
                    label_predict = predict_label.copy()
                    best_stump['dimension'] = i
                    best_stump['threshold'] = threshold
                    best_stump['inequal'] = inequal
    # output:
    # @ best_stump: a dictionary, save the decision tree info, i.e, dimension, threshold, inequal
    # @ min_error: a real value, the total error weights sum(w[i]), if y[i] != predict[i]
    # @ label_predict: a m*1 vector, predict each sample's label
    return best_stump, min_error, label_predict












