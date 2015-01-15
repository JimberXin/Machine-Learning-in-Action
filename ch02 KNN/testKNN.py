# ====================================================================
# Author: Junbo Xin
# Date: 2015/01/15
# Description:  test file for KNN.py

# ====================================================================


import KNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

data_set, labels = KNN.create_data_set()

test1 = array([1.2, 1.0])
test2 = array([0.1, 0.3])
k = 3
output_label1 = KNN.knn_classify(test1, data_set, labels, k)
output_label2 = KNN.knn_classify(test2, data_set, labels, k)
print test1, output_label1
print test2, output_label2


dating_mat, dating_label = KNN.file_to_matrix('datingTestSet2.txt')
for i in range(30):
    print dating_mat[i], dating_label[i]



fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dating_mat[:, 0], dating_mat[:, 1],
           15.0 * array(dating_label), 15.0 * array(dating_label))
plt.show()
