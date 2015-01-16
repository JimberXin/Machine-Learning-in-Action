# ====================================================================
# Author: Junbo Xin
# Date: 2015/01/15
# Description:  test file for KNN.py

# ====================================================================

import KNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def test_simple():
    data_set, labels = KNN.create_data_set()

    test1 = array([1.2, 1.0])
    test2 = array([0.1, 0.3])
    k = 3
    output_label1 = KNN.knn_classify(test1, data_set, labels, k)
    output_label2 = KNN.knn_classify(test2, data_set, labels, k)
    print test1, output_label1
    print test2, output_label2


def test_non_norm():
    dating_mat, dating_label = KNN.file_to_matrix('datingTestSet2.txt')
    for i in range(30):
        print dating_mat[i], dating_label[i]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_mat[:, 0], dating_mat[:, 1],
               15.0 * array(dating_label), 15.0 * array(dating_label))
    plt.show()


def date_class_test():
    ratio = 0.04    # ratio of the test examples
    # data_set:1000*3,  data_labels: 1000*1
    data_set, data_labels = KNN.file_to_matrix('datingTestSet2.txt')

    # normilize the data_set.   Note:  data_labels is not nessary to normlize
    norm_set, ranges, min_val = KNN.normalize(data_set)

    all_rows = norm_set.shape[0]   # number of all rows
    test_rows = int(ratio * all_rows)  # number of test rows
    error_num = 0
    for i in range(test_rows):
        # return the predict labels
        label_res = KNN.knn_classify(norm_set[i, :], norm_set[test_rows: all_rows, :],\
                                     data_labels[test_rows: all_rows, :], 3)
        print 'Classifier predict: %d, real result is: %d' % (label_res, data_labels[i])
        if label_res != data_labels[i]:
            error_num += 1
    print 'total error rate is: %f ' % (error_num * 1.0 / float(test_rows))


if __name__ == '__main__':
    # date_class_test()
    KNN.hand_writings_test()


