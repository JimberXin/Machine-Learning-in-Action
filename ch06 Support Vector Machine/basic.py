# =======================================================================
# @Author: Junbo Xin
# @Date: 2015/01/24
# @Description: basic funtion to be called by SVM
# =======================================================================

from numpy import *


# open a file: 1st and 2nd cols are features, 3rd col is label
def load_data_set(file_name):
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


# Giving the index of alpha: i. select one number(!=i) in (0, m)
def select_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


# make sure low <= aj <= high
def adjust_alpha(aj, high, low):
    if aj > high:
        aj = high
    if low > aj:
        aj = low
    return aj








