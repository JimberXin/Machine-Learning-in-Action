# =================================================================================================
# @Author: Junbo Xin
# @Date: 2015/02/04-06
# @Description: Test file for Regression
# =================================================================================================


import regression
import predictAge
import stageRidge
import ridgeRegress


def test_basic():
    x_arr, y_arr = regression.load_data_set('ex0.txt')
    w = regression.calc_weight(x_arr, y_arr)
    print w


def test_plot():
    regression.plot_line()


def test_plot_weight():
    x_arr, y_arr = regression.load_data_set('ex0.txt')
    regression.plot_local_weight([1.0, 0.5], x_arr, k=0.1)


def test_local_lr():
    regression.plot_line_lw()


def ridge_test():
    ridge_test()


def stage_test():
    stageRidge.test_stage()


def main():
    # test_basic()
    # test_plot()
    # test_local_lr()
    # test_plot_weight()
    # predictAge.train_lr()
    # ridgeRegress.ridge_test()
    stage_test()


if __name__ == '__main__':
    main()