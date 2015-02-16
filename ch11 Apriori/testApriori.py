# ====================================================================================
# Author: Junbo Xin
# Date: 2015/02/13-16
# Description:  test file for Apriori Algorithm
# ====================================================================================

from numpy import *
import Apriori


def test_create():
    data_set = Apriori.load_data_set()
    c1 = Apriori.create_c1(data_set)
    all_set = map(set, data_set)
    list_set, support_data = Apriori.scan_d(all_set, c1, 0.5)
    set2 = Apriori.generate_ck(list_set, 2)
    print set2


def test_apriori():
    data_set = Apriori.load_data_set()
    l, support_data = Apriori.apriori(data_set, 0.5)
    print l
    print support_data


def test_generate_rules():
    data_set = Apriori.load_data_set()
    l, support_data = Apriori.apriori(data_set, 0.5)
    print '='*100
    print l
    rules = Apriori.generate_rules(l, support_data, 0.5)
    print rules


def main():
    # test_create()
    # test_apriori()
    test_generate_rules()


if __name__ == '__main__':
    main()