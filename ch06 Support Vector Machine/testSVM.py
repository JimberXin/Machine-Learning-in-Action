# ======================================================================
# @Author: Junbo Xin
# @Date: 2015/01/25
# @Description: test file for SVM
# =======================================================================

import basic


def test_load():
    data_arr, label_arr = basic.load_data_set('testSet.txt')
    print label_arr


def main():
    test_load()


if __name__ == '__main__':
    main()


