# ====================================================================
# @Author: Junbo Xin
# @Date: 2015/01/20
# @Description: Test of Logistic Regressions
# ====================================================================

import logRegress
from numpy import *


def test_base():
    data_mat, label_mat = logRegress.load_data_set()
    w = logRegress.grad_ascent(data_mat, label_mat)
    logRegress.plot_fig(w)


def test_sto():
    data_arr, label_mat = logRegress.load_data_set()
    w = logRegress.sto_grad_ascent(array(data_arr), label_mat)
    logRegress.plot_fig(w)


def test_sto_improve():
    data_arr, label_mat = logRegress.load_data_set()
    w = logRegress.sto_grad_ascent_improve(array(data_arr), label_mat)
    logRegress.plot_fig(w)


def main():
    # test_base()
    # test_sto()
    test_sto_improve()


if __name__ == '__main__':
    main()

