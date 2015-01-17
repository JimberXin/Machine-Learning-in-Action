# =============================================================================
# Author: Junbo Xin
# Date: 2015/01/16
# Description:  Decision Tree

# =============================================================================

from math import log
import operator


# Simple function to create a data set
def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    feature_labels = ['no surfacing', 'flippers']   # correspond to 1st and 2nd column
    return data_set, feature_labels


# giving the data_set(last col is labels), calculate the total Shannon entropy
def calc_shannon_entropy(data_set):
    num = len(data_set)
    label_counts = {}
    for line in data_set:
        label = line[-1]  # get the label in the last col of data_set
        if label not in label_counts.keys():
            label_counts[label] = 0
        label_counts[label] += 1
    entropy = 0.0
    for key in label_counts.keys():
        prob = float(label_counts[key]) / num
        entropy -= prob * log(prob, 2)
    return entropy


# return a new data_set without having the current feature
def split_data_set(data_set, feature, value):
    ret_data_set = []
    for line in data_set:
        # if and only if the feature==value
        if line[feature] == value:
            reduce_vec = line[:feature]  # cut before feature
            reduce_vec.extend(line[feature+1:])  # cut after feature
            ret_data_set.append(reduce_vec)  # now the new list has all features except itself
    return ret_data_set


# select the features that has largest entropy to be the current choice
# data_set's last col are labels
def choose_features(data_set):
    num_of_features = len(data_set[0]) - 1    # feature numbers
    base_entropy = calc_shannon_entropy(data_set)   # label entropy
    best_gain = 0.0
    best_feature = -1
    for i in range(num_of_features):
        feature_list = [line[i] for line in data_set]
        unique_val = set(feature_list)  # get all the values of the current feature
        feature_entropy = 0.0
        for value in unique_val:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            feature_entropy += prob * calc_shannon_entropy(sub_data_set)
        current_info_gain = base_entropy - feature_entropy
        if current_info_gain > best_gain:
            best_gain = current_info_gain
            best_feature = i
    return best_feature


# if feature cannot split the label, then choose the label that is voted most
def majority_count(label_list):
    label_count = {}
    for line in label_list:
        if line not in label_count:
            label_count[line] = 0
        label_count[line] += 1
    sorted_label = sorted(label_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_label[0][0]


#
def create_tree(data_set, feature_labels):
    # step 1:  get the label list
    label_list = [line[-1] for line in data_set]
    # stop condition 1: If all labels are the same, just return it
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # stop condition 2: If there are no more features to split the labels
    #                   select the label that is voted most
    if len(data_set) == 1:
        return majority_count(label_list)

    # step 2: From all the features, select the one that has most info-gain
    best_feature = choose_features(data_set)
    best_feature_label = feature_labels[best_feature]

    # final result: save in ret_tree
    ret_tree = {best_feature_label: {}}
    del(feature_labels[best_feature])
    feature_values = [line[best_feature] for line in data_set]
    unique_val = set(feature_values)
    for value in unique_val:
        copy_labels = feature_labels[:]
        ret_tree[best_feature_label][value] = create_tree(
            split_data_set(data_set, best_feature, value), copy_labels)

    return ret_tree


def classify(input_tree, feature_labels, test_vec):
    fisrt_str = input_tree.keys()[0]
    second_str = input_tree[fisrt_str]
    feature_index = feature_labels.index(fisrt_str)   # turn to vector
    for key in second_str.keys():
        if test_vec[feature_index] == key:
            if type(second_str[key]).__name__ == 'dict':
                class_label = classify(second_str[key], feature_labels, test_vec)
            else:
                class_label = second_str[key]
    return class_label


def store_tree(input_tree, file_name):
    import pickle
    fw = open(file_name, 'w')
    pickle.dump(input_tree, fw)
    fw.close()


def load_tree(file_name):
    import pickle
    fr = open(file_name, 'r')
    return pickle.load(fr)

