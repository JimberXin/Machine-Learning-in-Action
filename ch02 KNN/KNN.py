# ====================================================================
# Author: Junbo Xin
# Date: 2015/01/15
# KNN:  K Nearest Neighbours
# Input: @new_input:   test example, a vector of size 1 * d
#        @data_set:  training examples, an array of size M * d
#        @labels: labels of data_set, a vector of size M * 1
#        @k:  numbers of neighbours to choose the class

# ====================================================================
from numpy import *
import operator


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def knn_classify(new_input, data_set, labels, k):
    data_size = data_set.shape[0]   # shape[0] is the row of the data set

    # step 1: calculate the Euclidean distance
    # tile(A, reps): construct an array by repeating A reps times
    # new_input: 1 * d;
    diff = tile(new_input, (data_size, 1)) - data_set
    square_diff = diff ** 2   # square error
    square_sum = sum(square_diff, axis=1)  # sum is perform by row
    distance = square_sum ** 0.5

    # step 2: sort the distance, in a ascending order
    # argsort() return the indices to sort the array
    sorted_dist = distance.argsort()

    class_count = {}  # initialize an empty dictionary
    for i in range(k):
        # step 3: choose the min k distance
        vote_label = labels[sorted_dist[i]]

        # step 4: counts the labels of different classes
        # dict.get(key, default=None): if exist, returns value of key, else return default
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # step 5: finds the maximum value: which class votes most
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]  # return the class that votes most


# open file file_name, return the data_set(n*3), and label vector(n*1)
def file_to_matrix(file_name):
    f = open(file_name)
    data_set = []
    label = []
    # not using readlines(), read the file a line each time
    for each_line in f:
        line = each_line.strip()
        line = line.split('\t')
        data_set.append(line[0:len(line)-1])
        label.append(int(line[-1]))
    return array(data_set), label


def normalize(data_set):
    # data_set: m*3;  min_val,max_val: 1*3
    min_val = data_set.min(0)  # find the min of the col (1 for the row)
    max_val = data_set.max(0)
    ranges = max_val - min_val
    norm_data_set = zeros(shape(data_set))
    row = data_set.shape[0]  # number of rows
    diff = data_set - tile(min_val, (row, 1))
    norm_set = norm_data_set/tile(ranges, (row, 1))
    return norm_set, ranges, min_val


