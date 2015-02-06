# =================================================================================================
# @Author: Junbo Xin
# @Date: 2015/02/04-06
# @Description: giving abalone.txt, predict age using regression algorithm
# =================================================================================================

import regression
from numpy import *


def res_error(y_arr, y_hat_arr):
    return ((y_arr - y_hat_arr)**2).sum()


def train_lr():
    train_x, train_y = regression.load_data_set('abalone.txt')
    # calculate the training y using the data set of the 0-100 samples
    y_train_hat1 = regression.local_weight_array_test(train_x[0:99], train_x[0:99], train_y[0:99], k=0.1)
    y_train_hat2 = regression.local_weight_array_test(train_x[0:99], train_x[0:99], train_y[0:99], k=1)
    y_train_hat3 = regression.local_weight_array_test(train_x[0:99], train_x[0:99], train_y[0:99], k=10)

    # calculate the training error and print
    y_train_error1 = res_error(train_y[0:99], y_train_hat1.T)
    y_train_error2 = res_error(train_y[0:99], y_train_hat2.T)
    y_train_error3 = res_error(train_y[0:99], y_train_hat3.T)
    print 'k = 0.1, training error is: ', y_train_error1
    print 'k = 1,   training error is: ', y_train_error2
    print 'k = 10,  training error is: ', y_train_error3

    print '=' * 50
    # calculate the test y using the data set of the 100-199 samples
    y_test_hat1 = regression.local_weight_array_test(train_x[100:199], train_x[0:99], train_y[0:99], k=0.1)
    y_test_hat2 = regression.local_weight_array_test(train_x[100:199], train_x[0:99], train_y[0:99], k=1)
    y_test_hat3 = regression.local_weight_array_test(train_x[100:199], train_x[0:99], train_y[0:99], k=10)

    # calculate the test error and print
    y_test_error1 = res_error(train_y[100:199], y_test_hat1.T)
    y_test_error2 = res_error(train_y[100:199], y_test_hat2.T)
    y_test_error3 = res_error(train_y[100:199], y_test_hat3.T)
    print 'k = 0.1, test error is: ', y_test_error1
    print 'k = 1,   test error is: ', y_test_error2
    print 'k = 10,  test error is: ', y_test_error3

