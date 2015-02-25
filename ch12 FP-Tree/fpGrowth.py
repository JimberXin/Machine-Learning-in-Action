# ====================================================================================
# Author: Junbo Xin
# Date: 2015/02/17-25
# Description:  Frequent Pattern Growth
# ====================================================================================


from numpy import *


# define the structure of the tree node
class TreeNode:
    def __init__(self, name_value, num_count, parent_node):
        self.name = name_value
        self.count = num_count
        self.node_link = None
        self.parent = parent_node
        self.children = {}

    def add(self, num_count):
        self.count += num_count

    def display(self, index=1):
        print ' '*index, self.name, ' ', self.count
        for child in self.children.values():
            child.display(index+1)


def load_simple_data():
    simple_dat = [['r', 'z', 'h', 'j', 'p'],
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                ['z'],
                ['r', 'x', 'n', 'o', 's'],
                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simple_dat


def create_init_set(data_set, mini_up=1):
    ret_dict = {}
    for trans in data_set:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


# ========================================= create FP tree ==================================
# giving the data_set(already initialize as a dict), create the FP tree
def create_tree(data_set, mini_up=1):
    head_table = {}
    print data_set
    for trans in data_set:
        for item in trans:
            print [trans]
            head_table[item] = head_table.get(item, 0) + data_set[trans]
            print '='*10, item, head_table[item]

    for k in head_table.keys():
        if head_table[k] < mini_up:
            del(head_table[k])

    # a set with unique items only
    freq_item_set = set(head_table.keys())

    # if the set has no elements, just return
    if len(freq_item_set) == 0:
        return None, None

    for k in head_table:
        head_table[k] = [head_table[k], None]

    ret_tree = TreeNode('Null Set', 1, None)

    print head_table

    # go through the set 2rd time
    for tran_set, count in data_set.items():
        local = {}
        print tran_set, '*'*2, count
        for item in tran_set:
            if item in freq_item_set:
                local[item] = head_table[item][0]  # 0 is the counts, 1 is

        print local

        if len(local) > 0:
            order_items = [v[0] for v in sorted(local.items(), key=lambda p:p[1], reverse=True)]
            update_tree(order_items, ret_tree, head_table, count)
    return ret_tree, head_table


#
def update_tree(items, tree, head_table, count):
    if items[0] in tree.children:
        tree.children[items[0]].add(count)
    else:
        tree.children[items[0]] = TreeNode(items[0], count, tree)

        if head_table[items[0]][1] == None:
            head_table[items[0]][1] = tree.children[items[0]]
        else:
            update_header(head_table[items[0]][1], tree.children[items[0]])

    if len(items) > 1:
        update_tree(items[1::], tree.children[items[0]], head_table, count)


# add target_node at the end of node_to_test
def update_header(node_to_test, target_node):
    while node_to_test.node_link is not None:
        node_to_test = node_to_test.node_link
    node_to_test.node_link = target_node


# ====================================== Mining frequent set ================================
def ascend_tree(leaf_node, pre_path):
    if leaf_node.parent is not None:
        pre_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, leaf_node)


# find the previous path giving the tree node
def find_prefix_path(base_path, tree_node):
    cond_pats = {}
    while tree_node is not None:
        pre_fix_path = []
        ascend_tree(tree_node, pre_fix_path)
        if len(pre_fix_path) > 1:
            cond_pats[frozenset(pre_fix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return cond_pats