# ======================================================================
# @Author: Junbo Xin
# @Date: 2015/01/25
# @Description: test file for SVM
# =======================================================================

import basic
import SVM_Platt
from numpy import *


def test_load():
    data_arr, label_arr = basic.load_data_set('testSet.txt')
    print label_arr


def test_smo_basic():
    data_arr, label_arr = basic.load_data_set('testSet.txt')
    b, alphas = basic.smo_basic(data_arr, label_arr, 0.6, 0.001, 40)
    print alphas[alphas>0]


def test_smo_platt():
    data_arr, label_arr = basic.load_data_set('testSet.txt')
    b, alphas = SVM_Platt.smo_platt(data_arr, label_arr, 0.6, 0.001, 40)
    print alphas[alphas>0]


def test_rbf():
    SVM_Platt.test_rbf()


def main():
    # test_smo_basic()
    # test_smo_platt()
    test_rbf()


if __name__ == '__main__':
    main()


