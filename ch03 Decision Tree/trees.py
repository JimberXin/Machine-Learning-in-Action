# =============================================================================
# Author: Junbo Xin
# Date: 2015/01/16
# Description:  decision tree
# Input: @new_input:   test example, a vector of size 1 * d
#        @data_set:  training examples, an array of size M * d
#        @labels: labels of data_set, a vector of size M * 1
#        @k:  numbers of neighbours to choose the class

# =============================================================================

from math import log


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


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


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
    base_entropy = calc_shannon_entropy(data_set)
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

