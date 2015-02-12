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
    regressionTree.plot_data_set('exp2.txt')


def test_prune():
    # training a tree
    data_arr = regressionTree.load_data_set('ex2.txt')
    training_mat = mat(data_arr)
    my_tree = regressionTree.create_tree(training_mat, ops=(0, 1))

    # test using the tree
    test_set = regressionTree.load_data_set('ex2test.txt')
    test_mat = mat(test_set)
    tree = regressionTree.prune(my_tree, test_mat)
    print tree


def test_model_tree():
    data_arr = regressionTree.load_data_set('exp2.txt')
    training_mat = mat(data_arr)
    my_tree = regressionTree.create_tree(training_mat, regressionTree.model_leaf, \
                                         regressionTree.model_err, ops=(1, 10))
    print my_tree
    regressionTree.plot_data_set('exp2.txt')



def main():
    # test_basic()
    # test_create_tree()
    # test_plot()
    # test_prune()
    # test_model_tree()
    regressionTree.compare_regression()


if __name__ == '__main__':
    main()
