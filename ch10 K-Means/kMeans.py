# ===============================================================================================
# Author: Junbo Xin
# Date: 2015/02/12
# Description:  K Means Algorithm
# ===============================================================================================
from numpy import *


def load_data_set(file_name):
    data_mat = []
    fr = open(file_name).readlines()
    for line in fr:
        cur_line = line.strip().split('\t')
        float_line = map(float, cur_line)  # turn the string to float format
        data_mat.append(float_line)
    return data_mat


# giving two vec, calculate the distance(Eclud)
def calc_dist(vec1, vec2):
    return sqrt(sum(power(vec1-vec2, 2)))


# return randomly k center: a k * n matrix
def rand_center(data_set, k):
    n = shape(data_set)[1]
    center = mat(zeros((k ,n)))
    for i in range(n):
        min_val = min(data_set[:, i])
        range_val = float(max(data_set[:, i]) - min_val)
        center[:, i] = min_val + range_val*random.rand(k, 1)
    return center


def k_means(data_set, k, dist_fun=calc_dist, cen_init=rand_center):
    m, n = shape(data_set)
    # 1st col save the index, 2nd col save the error of current data and center
    cluster = mat(zeros((m, 2)))

    # step 1: initialize k center
    center = cen_init(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False

        # step 2: go through each sample
        for i in range(m):
            min_dist = inf
            min_index = -1

            # step 3: go through each center j(j=1,2,..,k), calculate the distance
            for j in range(k):
                dist = dist_fun(data_set[i, :], center[j, :])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            if cluster[i, 0] != min_index:
                cluster_changed = True
            cluster[i, :] = min_index, min_dist**2

        # step 4: update the k centers
        # print center
        for cen in range(k):
            # nonzero returns a 2d matrix, rows and cols that are nonzero
            update_cluster = data_set[nonzero(cluster[:, 0].A == cen)[0]]  # .A turns to array
            center[cen, :] = mean(update_cluster, axis=0)  # axis means calculate the col

    '''
    print '='*40
    print 'final center is:'
    print center
    print cluster
    '''

    return center, cluster


def plot_k_means(data_set, cluster, center, k):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # b:blue  y:yellow  r:red  c:cyan  k:black  w:white

    # plot the figure, at most k=8
    color = ['y', 'r', 'm', 'b', 'k', 'c', 'w', 'g']
    marker = ['s', 'o', '^', 'p', 'v', '>', '<', 'd']
    for i in range(k):
        data_mat = data_set[nonzero(cluster[:, 0].A == i)[0]]
        ax.scatter(array(data_mat[:, 0]), array(data_mat[:, 1]), \
                   s=70, marker=marker[i], c=color[i], label=str(i+1))
        ax.scatter(center[i, 0], center[i, 1], s=200, marker='h', c='g')
    plt.legend()
    plt.show()


def binary_k_means(data_set, k, dist_fun=calc_dist):
    m = shape(data_set)[0]
    cluster = mat(zeros((m, 2)))

    # step 1: first create only one center
    center = mean(data_set, axis=0).tolist()[0]
    center_list = [center]
    for i in range(m):
        cluster[i, 1] = dist_fun(mat(center), data_set[i, :])**2

    while len(center_list) < k:
        min_err = inf

        # step 2: for each center, try to split it, find the one that has least error
        for i in range(len(center_list)):
            current_cluster = data_set[nonzero(cluster[:, 0].A == i)[0], :]  # the i-th center
            center_mat, cluster_mat = k_means(current_cluster, 2, dist_fun)
            err_split = sum(cluster_mat[:, 1])
            err_non_split = sum(cluster[nonzero(cluster[:, 0].A != i)[0], 1])
            print 'split error, non split error is: ', err_split, err_non_split

            if (err_split + err_non_split) < min_err:
                best_center_split = i
                best_new_center = center_mat
                best_cluster = cluster_mat.copy()
                min_err = err_split + err_non_split

        # step 3: update the new allocation result, k=2, so there's 0 and 1 clusters only
        best_cluster[nonzero(best_cluster[:, 0].A == 1)[0], 0] = len(center_list)
        best_cluster[nonzero(best_cluster[:, 0].A == 0)[0], 0] = best_center_split

        print 'the best center to split is:', best_center_split
        print 'the len of best cluster is:', len(best_cluster)

        # for the '0' class, update directly in the center, for the '1' class, append at the end
        center_list[best_center_split] = best_new_center[0, :].tolist()[0]
        center_list.append(best_new_center[1, :].tolist()[0])

        '''
        print '='*60
        print cluster
        print '*'*60
        print best_cluster
        '''
        cluster[nonzero(cluster[:, 0].A == best_center_split)[0], :] = best_cluster

    return mat(center_list), cluster










