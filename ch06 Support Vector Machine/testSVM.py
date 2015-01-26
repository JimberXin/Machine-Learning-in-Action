# ======================================================================
# @Author: Junbo Xin
# @Date: 2015/01/25
# @Description: test file for SVM
# =======================================================================

import basic


def test_load():
    data_arr, label_arr = basic.load_data_set('testSet.txt')
    print label_arr


def test_smo_basic():
    data_arr, label_arr = basic.load_data_set('testSet.txt')
    b, alphas = basic.smo_basic(data_arr, label_arr, 0.6, 0.001, 40)
    print alphas[alphas>0]


def main():
    test_smo_basic()


if __name__ == '__main__':
    main()


