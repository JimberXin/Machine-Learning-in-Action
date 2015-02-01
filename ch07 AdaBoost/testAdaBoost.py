# =================================================================================================
# @Author: Junbo Xin
# @Date: 2015/01/29-30
# @Description: Adaboost algorithm
# =================================================================================================

from numpy import *
import adaBoost


def test_stump():
    data_mat, data_label = adaBoost.load_simple_data()
    D = mat(ones((5, 1))/5)
    # adaBoost.build_stump(data_mat, data_label, D)
    classifier = adaBoost.ada_boost_train(data_mat, data_label, 9)


def main():
    # test_stump()
    data_mat, data_label = adaBoost.load_simple_data()
    D = mat(ones((5, 1))/5)
    adaBoost.ada_boost_train(data_mat, data_label)

if __name__ == '__main__':
    main()
