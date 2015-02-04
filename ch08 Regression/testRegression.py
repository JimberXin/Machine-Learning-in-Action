from numpy import *
import regression


def test_basic():
    x_arr, y_arr = regression.load_data_set('ex0.txt')
    w = regression.calc_weight(x_arr, y_arr)
    print w


def test_plot():
    regression.plot_line()


def main():
    # test_basic()
    test_plot()


if __name__ == '__main__':
    main()