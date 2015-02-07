# =============================================================================
# Author: Junbo Xin
# Date: 2015/02/07
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


def main():
    test_basic()


if __name__ == '__main__':
    main()
