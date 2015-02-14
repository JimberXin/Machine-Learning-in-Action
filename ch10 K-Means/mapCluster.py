# ===============================================================================================
# Author: Junbo Xin
# Date: 2015/02/12
# Description:  K Means Algorithm
# ===============================================================================================

from numpy import *
import kMeans
import matplotlib
import matplotlib.pyplot as plt


# calculate the distance of two points on the earth
def calc_earth_dist(vec1, vec2):
    a = sin(vec1[0, 1]*pi/180) * sin(vec2[0, 1]*pi/180)
    b = cos(vec1[0, 1]*pi/180) * cos(vec2[0, 1]*pi/180) *\
                    cos(pi*(vec2[0, 0]-vec1[0, 0])/180)

    return arccos(a+b)*6371.0


def cluster_clubs(num_cluster=5):
    data_list = []
    for line in open('places.txt').readlines():
        line_arr = line.split('\t')
        # the 4th and 5th col of the file is the longitude and latitude
        data_list.append([float(line_arr[4]), float(line_arr[3])])
    data_mat = mat(data_list)
    center, cluster = kMeans.binary_k_means(data_mat, num_cluster, dist_fun=calc_earth_dist)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    colors = ['k', 'r', 'b', 'c', 'g']
    markers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    ax_props = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **ax_props)

    img = plt.imread('Portland.png')
    ax0.imshow(img)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)

    # for each cluster, plot the scatters
    for i in range(num_cluster):
        current_clu = data_mat[nonzero(cluster[:, 0].A == i)[0], :]
        marker_sty = markers[i % len(markers)]
        color_sty = colors[i % len(colors)]
        ax1.scatter(current_clu[:, 0].flatten().A[0],
                    current_clu[:, 1].flatten().A[0], marker=marker_sty, color=color_sty, s=90)
    ax1.scatter(center[:, 0].flatten().A[0],
                center[:, 1].flatten().A[0], marker='+', color='g', s=300)
    plt.show()

