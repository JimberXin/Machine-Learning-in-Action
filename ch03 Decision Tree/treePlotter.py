# ===================================================================================
# Author: Junbo Xin
# Date: 2015/01/17
# Description:  Plot the figure of Decision Tree

# ===================================================================================

import matplotlib.pyplot as plt


# define the basic configure: dicision_node as middle node and leaf_node as end node
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# Using annotations of Matplotlib to draw fig
def plot_node(node_text, center_point, parent_point, node_type):
    create_plot.axl.annotate(node_text, xy=parent_point, xycoords='axes fraction',
            xytext=center_point, textcoords='axes fraction',
            va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


'''
def create_plot():
    fig = plt.figure(1, facecolor='green')
    fig.clf()
    create_plot.axl = plt.subplot(111, frameon=False)
    plot_node('Decision Node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('Leaf Node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()
'''

# giving the Decision Tree: tree, return it's total number of leaves
def get_num_of_leafs(tree):
    num_of_leafs = 0
    first_str = tree.keys()[0]
    second_str = tree[first_str]
    for key in second_str.keys():
        # if current element is still a dictionary,then call the function itself
        if type(second_str[key]).__name__ == 'dict':
            num_of_leafs += get_num_of_leafs(second_str[key])
        else:
            num_of_leafs += 1   # otherwise it's a leaf node, add the num
    return num_of_leafs

# giving the Decision Tree: tree, return it's maximum depth from root to leaf
def get_tree_depth(tree):
    max_depth = 0
    first_str = tree.keys()[0]
    second_str = tree[first_str]
    for key in second_str.keys():
        # if current element is still a dictionary,then call the function itself
        if type(second_str[key]).__name__ == 'dict':
            cur_depth = 1 + get_tree_depth(second_str[key])
        else:
            cur_depth = 1   # if it's leaf node, depth is 1
        if cur_depth > max_depth:
            max_depth = cur_depth
    return max_depth


def retrive_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return list_of_trees[i]


def plot_mid_text(center_pt, parent_pt, text_str):
    x_axis = (parent_pt[0] - center_pt[0])/2.0 + center_pt[0]
    y_axis = (parent_pt[1] - center_pt[1])/2.0 + center_pt[1]
    create_plot.axl.text(x_axis, y_axis, text_str, va="center", ha="center", rotation=30)


def plot_tree(decision_tree, parent_pt, node_text):
    #  step 1: calculate the width and depth of the tree
    num_of_leafs = get_num_of_leafs(decision_tree)
    depth = get_tree_depth(decision_tree)

    # step 2: calculate the current center point
    first_str = decision_tree.keys()[0]
    center_pt = (plot_tree.xOff + (1.0 + float(num_of_leafs))/2.0/plot_tree.totalW, plot_tree.yOff)

    # step 3: plot the line between current_pt and parent_pt
    plot_mid_text(center_pt, parent_pt, node_text)

    # step 4: fill the text of current label
    plot_node(first_str, center_pt, parent_pt, decision_node)

    # step 5: coninue plotting the tree
    second_str = decision_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD   # decrease, go down
    for key in second_str.keys():
        if type(second_str[key]).__name__ == 'dict':
            plot_tree(second_str[key], center_pt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
            # now xOff and yOff is current leaf node's parent
            plot_node(second_str[key], (plot_tree.xOff, plot_tree.yOff), center_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), center_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD   # don't forget to go  up


# main function to create the tree
def create_plot(in_tree):
    fig = plt.figure(1, facecolor='gray')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.axl = plt.subplot(111, frameon=False, **axprops)  # need to know further
    plot_tree.totalW = float(get_num_of_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))

    # trace the node that has been plot: xOff, yOff
    plot_tree.xOff = -0.5/plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')    # start plot the tree, from top to down
    plt.show()






