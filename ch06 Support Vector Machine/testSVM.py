# ======================================================================
# @Author: Junbo Xin
# @Date: 2015/01/25
# @Description: test file for SVM
# =======================================================================

import basic
import SVM_Platt
from numpy import *
import classifyDigits


def test_load():
    data_arr, label_arr = basic.load_data_set('testSet.txt')
    print label_arr


def test_smo_basic():
    data_arr, label_arr = basic.load_data_set('testSet.txt')
    svm = basic.smo_basic(data_arr, label_arr, 0.6, 0.001, 40)
    print svm.alphas[svm.alphas > 0]


def test_smo_platt():
    data_arr, label_arr = basic.load_data_set('testSet.txt')
    svm = SVM_Platt.train_svm(data_arr, label_arr, 0.6, 0.001, 40)
    print svm.alphas[svm.alphas>0]


def test_svm():
    data_arr, label_arr = basic.load_data_set('testSet.txt')
    svm = SVM_Platt.train_svm(data_arr, label_arr, 200, 0.001, 10000, ('lin', 0))
    # test_x, test_y = basic.load_data_set('testSetRBF2.txt')
    SVM_Platt.test_svm(svm, data_arr, label_arr)
    SVM_Platt.show_svm(svm)


def test_digits():
    classifyDigits.test_digits_svm()


def main():
    # test_smo_basic()
    # test_smo_platt()
    test_svm()
    # test_digits()

if __name__ == '__main__':
    main()


