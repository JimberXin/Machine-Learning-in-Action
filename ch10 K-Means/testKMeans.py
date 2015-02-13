# ===============================================================================================
# Author: Junbo Xin
# Date: 2015/02/12
# Description:  Test file for K Means
# ===============================================================================================
import kMeans
import mapCluster
from numpy import *


def test_load():
    data_mat = mat(kMeans.load_data_set('testSet.txt'))
    print min(data_mat[:, 0])
    print max(data_mat[:, 1])
    print kMeans.rand_center(data_mat, 2)


def test_k_means():
    k = 4
    data_mat = mat(kMeans.load_data_set('testSet.txt'))
    center, cluster = kMeans.k_means(data_mat, k)
    kMeans.plot_k_means(data_mat, cluster, center, k)


def test_binary_k_means():
    k = 3
    data_mat = mat(kMeans.load_data_set('testSet2.txt'))
    center, cluster = kMeans.binary_k_means(data_mat, k)
    kMeans.plot_k_means(data_mat, cluster, center, k)


def test_map_cluster():
    mapCluster.cluster_clubs(5)


def main():
    # test_load()
    # test_k_means()
    # test_binary_k_means()
    test_map_cluster()


if __name__ == '__main__':
    main()