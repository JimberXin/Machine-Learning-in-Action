__author__ = 'JimberXin'

import KNN
from numpy import *

data_set, labels = KNN.create_data_set()

test1 = array([1.2, 1.0])
test2 = array([0.1, 0.3])
k = 3
output_label1 = KNN.knn_classify(test1, data_set, labels, k)
output_label2 = KNN.knn_classify(test2, data_set, labels, k)
print test1, output_label1
print test2, output_label2


