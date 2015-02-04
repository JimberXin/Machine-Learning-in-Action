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
    m, n = shape(data_mat)
    ret_arr = ones((m, 1))  # the whole vector initializes with 1
    if inequal == 'less than':
        ret_arr[data_mat[:, dimen] <= threshold] = -1.0
    else:
        ret_arr[data_mat[:, dimen] > threshold] = -1.0
    return ret_arr


# giving the training set, return the current best stump classifier(2-class classifier)
def build_stump(data_arr, label_arr, weight_arr):
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).T
    m, n = shape(data_mat)
    num_step = 10.0
    best_stump = {}   # decision tree info
    label_predict = mat(zeros((m, 1)))   # a m*1 vector, 1 or -1
    min_error = inf

    # min_arr, max_arr saves the min and max value of each col, 0 stands for perform by col
    min_arr = data_mat.min(0)
    max_arr = data_mat.max(0)

    #  step 1: go through each dimension of data set, i.e, each col
    for i in range(n):
        # min_val = data_mat[:, i].min()
        # max_val = data_mat[:, i].max()
        step_size = (max_arr[0, i] - min_arr[0, i]) / num_step
        # step 2: go through each threshold of this dimension
        for j in range(-1, int(num_step)+1):

            # step 3: go through both equality: <  and  >
            for inequal in ['less than', 'greater than']:
                threshold = min_arr[0, i] + float(j)*step_size
                predict_label = stump_classify(data_mat, i, threshold, inequal)
                error_arr = mat(ones((m, 1)))
                error_arr[predict_label == label_mat] = 0
                weight_error = weight_arr.T * error_arr
                '''
                print 'dimension: %d, threshold: %.2f, inequal: %s, weight error: %f' %\
                      (i, threshold, inequal, weight_error)
                '''

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


# giving the data set, return the classifier that is consist of several weak classifiers
def ada_boost_train(data_arr, label_arr, num_iter=40):
    weak_classifier = []
    m, n = shape(data_arr)
    weight = mat(ones((m, 1))/m)
    class_weight = mat(zeros((m, 1)))

    for i in range(num_iter):
        best_stump, error, label_predict = build_stump(data_arr, label_arr, weight)
        # print 'weights:', weight.T
        # in case error is 0, set the divisor to be a small value near 0
        alpha = float(0.5 * log((1.0-error)/max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_classifier.append(best_stump)
        # print 'label predict: ', label_predict.T

        # weights of the sample to be update
        expon = multiply(-1*alpha*mat(label_arr).T, label_predict)
        weight = multiply(weight, exp(expon))

        # update the weights of each sample in the next iteration
        # Zm = weight.sum()
        weight = weight / weight.sum()    # w(m+1) = w(m)*exp[-alpha(m)*y(i)*Gm(xi)]

        # obtain the i-th iteration linear classifier:  f(x) = sum(m=1,...,i)[alpha[i]*Gm(x)]
        class_weight += alpha * label_predict
        # print 'class estimate: ', class_weight.T

        errors = multiply(sign(class_weight) != mat(label_arr).T, ones((m, 1)))
        error_rate = errors.sum()/m
        print 'total error: ', error_rate
        if error_rate == 0.0:
            break

    # return a list, each element is a dict includes:
    #         dimension d, inequality > or <, threshold, and alpha
    return weak_classifier, class_weight  #


# giving the data set and losts of single classifier, get the combination of single classifier
def ada_classifier(data, weak_classifier):
    data_mat = mat(data)
    m, n = shape(data_mat)
    final_classifier = mat(zeros((m, 1)))
    for i in range(len(weak_classifier)):
        classifier = stump_classify(data_mat, weak_classifier[i]['dimension'], \
                                    weak_classifier[i]['threshold'], weak_classifier[i]['inequal'])

        final_classifier += weak_classifier[i]['alpha'] * classifier
        # print final_classifier
    return sign(final_classifier)














