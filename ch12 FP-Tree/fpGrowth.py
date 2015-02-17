# ====================================================================================
# Author: Junbo Xin
# Date: 2015/02/17
# Description:  Frequent Pattern Growth
# ====================================================================================


from numpy import *


def load_simple_data():
    simple_dat = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simple_dat


def create_tree(data_set, mini_up=1):
    ret_dict = {}
    for trans in data_set:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


