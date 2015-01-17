# =============================================================================
# Author: Junbo Xin
# Date: 2015/01/18
# Description:  Input:  @data_set,  @feature_labels
#               Output: @Decision Tree:  my_tree
#    1) test_base(): use the simple data:
"""
   data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
"""
#    2) test_grass(): use the 'lenses.txt'
# =============================================================================

import trees
import treePlotter


def test_base():
    data_set, labels = trees.create_data_set()
    my_tree = trees.create_tree(data_set, labels)
    trees.store_tree(my_tree, 'classifierStorage.txt')
    load_tree = trees.load_tree('classifierStorage.txt')
    print load_tree


def test_glass():
    fr = open('lenses.txt')
    lenses = [line.strip().split('\t') for line in fr.readlines()]
    lense_labels = ['age', 'precript', 'astigmatic', 'tearRate']
    lense_tree = trees.create_tree(lenses, lense_labels)
    treePlotter.create_plot(lense_tree)


if __name__ == '__main__':
    test_base()
    test_glass()





