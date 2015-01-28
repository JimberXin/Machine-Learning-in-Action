# ========================================================================================================
# @Author: Junbo Xin
# @Date: 2015/01/28
# @Description: Using SVM binary classifier to classify handwriting digit in KNN, only predict is 9 or not
# ========================================================================================================

from numpy import *
from basic import *
import SVM_Platt


# transfer the '0-1' txt image to a 1*1024 matrix and return it
def img_to_vec(file_name):
    fr = open(file_name)
    rows = 32
    cols = 32
    vec = zeros((1, 1024))
    for row in range(rows):
        line = fr.readline()
        for col in range(cols):
            vec[0, row*cols+col] = int(line[col])
    return vec


def load_images(dir_name):
    from os import listdir
    real_labels = []

    # list all the file name in the directory
    training_list = listdir(dir_name)
    training_num = len(training_list)
    training_mat = zeros((training_num, 1024))  # each row is a 1*1024 sample
    for row in range(training_num):
        file_name = training_list[row]
        file_name_str = file_name.split('.')[0]
        real_digit = int(file_name_str.split('_')[0])

        # 2-class classifier(1 and -1), if digit is 9, label is 1, otherwise label is -1
        if real_digit == 9:
            real_labels.append(1)
        else:
            real_labels.append(-1)
        training_mat[row, :] = img_to_vec('%s/%s' % (dir_name, file_name))  # add training set
    return training_mat, real_labels


def test_digits_svm():
    #
    # decision boundary: y = sum(y[i] * alphas[i] * K(x,x[i]) + b
    train_x, train_y = load_images('digits/trainingDigits')
    kernel = ('rbf', 2.3)
    svm = SVM_Platt.train_svm(train_x, train_y, 200, 0.0001, 10000, kernel)

    # calculate training error rate
    # SVM_Platt.test_svm(svm, train_x, train_y)

    # calculate test error rate
    test_x, test_y = load_images('digits/testDigits')
    SVM_Platt.train_svm(svm, test_x, test_y)
    SVM_Platt.show_svm(svm)




