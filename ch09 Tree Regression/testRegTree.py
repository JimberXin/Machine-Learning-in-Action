# =============================================================================
# Author: Junbo Xin
# Date: 2015/02/07-10
# Description:  Regression Tree
# =============================================================================

from numpy import *
import regressionTree


def test_basic():
    test_mat = mat(eye(4))
    print test_mat
    mat0, mat1 = regressionTree.bin_split_data(test_mat, 1, 0.5)
    print mat0
    print mat1


def test_create_tree():
    data_set = regressionTree.load_data_set('ex0.txt')
    data_mat = mat(data_set)
    tree = regressionTree.create_tree(data_mat)
    print tree


def test_plot():
    regressionTree.plot_data_set('ex0.txt')


def main():
    # test_basic()
    # test_create_tree()
    test_plot()


if __name__ == '__main__':
    main()
