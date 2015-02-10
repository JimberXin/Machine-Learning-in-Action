# =============================================================================
# Author: Junbo Xin
# Date: 2015/02/07-09
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
    my_dat = regressionTree.load_data_set('ex00.txt')
    my_mat = mat(my_dat)
    tree = regressionTree.create_tree(my_mat)
    print tree


def main():
    # test_basic()
    test_create_tree()


if __name__ == '__main__':
    main()
