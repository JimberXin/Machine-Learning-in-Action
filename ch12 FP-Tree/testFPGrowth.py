# ====================================================================================
# Author: Junbo Xin
# Date: 2015/02/17
# Description:  Frequent Pattern Growth
# ====================================================================================

from numpy import *
import fpGrowth


def test_simple():
    data_set = fpGrowth.load_simple_data()
    dic = fpGrowth.create_tree(data_set)
    print dic


def main():
    test_simple()


if __name__ == '__main__':
    main()