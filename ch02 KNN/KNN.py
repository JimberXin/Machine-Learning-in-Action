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
from os import listdir
from os import linesep


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# main classifier, args:
#     @training data: data_set, labels
#     @test data: new_input
#     @k:  nearest k neighbours
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
    # not using readlines(), but read the file a line each time, in case it's a big file
    for each_line in f:
        line = each_line.strip()
        line = line.split('\t')
        data_set.append([float(x) for x in line[0:len(line)-1]])  # convert string to num
        label.append(int(line[-1]))  # if not convert, it's a list of 1 * n
    return array(data_set), mat(label).T


def normalize(data_set):
    # data_set: m*3;  min_val,max_val: 1*3
    min_val = data_set.min(0)  # find the min of the col (1 for the row)
    max_val = data_set.max(0)
    ranges = max_val - min_val
    # norm_data_set = zeros(shape(data_set))
    row = data_set.shape[0]  # number of rows
    norm_data_set = data_set - tile(min_val, (row, 1))
    norm_set = norm_data_set/tile(ranges, (row, 1))
    return norm_set, ranges, min_val


# transfer the '0-1' txt image into a vector: 1*1024  and return it
def img_to_vector(file_name):
    fr = open(file_name)
    rows = 32
    cols = 32
    vec = zeros((1, 1024))
    # vec = [int(word) for line in fr for word in line.strip().split('\t')]
    # below is not pythonic at all!!!!
    for row in range(rows):
        line = fr.readline()
        for col in range(cols):
            vec[0, cols*row + col] = int(line[col])
    return vec


def hand_writings_test():
    """
    @Traing Input:  1934 txt file, named like: 3_114.txt
    @Test Input: 946 txt file
    :return:
    """
    real_labels = []
    # list all the file in the directory, save the file name in str format
    training_list = listdir('digits/trainingDigits')
    training_num = len(training_list)
    training_mat = zeros((training_num, 1024))  # training input for the classifier
    fw = open('hand_writings_ouput.txt', 'w')  # save the output result in the disk
    for row in range(training_num):
        file_name = training_list[row]   # get the file name ,like '2_110.txt'
        file_name_str = file_name.split('.')[0]   # get the file name without 'txt'
        real_digit = int(file_name_str.split('_')[0])  # get the real number
        real_labels.append(real_digit)
        training_mat[row, :] = img_to_vector('digits/trainingDigits/%s' % file_name)

    # test process
    test_list = listdir('digits/testDigits')
    error_count = 0
    test_num = len(test_list)
    for row in range(test_num):
        file_name = test_list[row]  # get the test file name
        file_name_str = file_name.split('.')[0]
        real_digit = int(file_name_str.split('_')[0])
        predict_vector = img_to_vector('digits/trainingDigits/%s' % file_name)
        predict_num = knn_classify(predict_vector, training_mat, real_labels, 3)
        # print 'Classifier predicts: %d, real digit is: %d ' % (predict_num, real_digit)
        output_str = 'Classifier predicts: %d, real digit is: %d' % (predict_num, real_digit)
        fw.write('%s%s' % (output_str, linesep))  # save the result with the end of th line: linesep
        if predict_num != real_digit:
            error_count += 1

    fw.close()
    print 'Total wrong predict counts is: %d of %d' % (error_count, test_num)
    print 'Total predict error rate is: %f ' % (error_count*1.0/float(test_num))







