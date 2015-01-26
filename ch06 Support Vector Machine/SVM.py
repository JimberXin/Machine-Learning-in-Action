from basic import *
from numpy import *


# using basic SMO
# Giving training data: data_input and label_input, and para C, toler, max_iter
def smo_basic(data_input, label_input, C, toler, max_iter):
    # step 0: initialization for data and label, alpha initialized with 0
    data_mat = mat(data_input)
    label_mat = mat(label_input).transpose()
    b = 0.0
    m, n = shape(data_mat)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        # step 1: select first alpha i
        alpha_pair_changed = 0
        for i in range(m):
            predict_i = float(multiply(alphas, label_mat).T * \
                             (data_mat * data_mat[i, :].T)) + b
            error_i = predict_i - float(label_mat[i])

            # step 2: if alpha i change a lot, go to optimization
            if (label_mat[i] * error_i < -toler and alphas[i] < C) or \
               (label_mat[i] * error_i > toler and alphas[i] > 0):
                j = select_rand(i, m)   # randomly select the second alpha j
                predict_j = float(multiply(alphas, label_mat).T * \
                                 (data_mat * data_mat[i, :].T)) + b
                error_j = predict_j - float(label_mat[j])

                # step 4: allocate new memory for alpha i and j, in case override
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # step 5: make sure alpha value: 0 <= alpha[k] <=C, k = 1,2,...,m
                if label_mat[i] != label_mat[j]:
                    Low = max(0, alphas[j]-alphas[i])
                    High = min(C, C+alphas[j]-alphas[i])
                else:
                    Low = max(0, alphas[j]+alphas[i]-C)
                    High = min(C, alphas[i]+alphas[j])
                if Low == High:
                    print 'Low==High'
                    continue

                # step 6: calculate eta = K11 + K22 - 2K12, make sure eta>0
                eta = data_mat[i, :]*data_mat[i, :].T + data_mat[j, :]*data_mat[j, :].T \
                      - 2.0*data_mat[i, :]*data_mat[j, :].T
                if eta <= 0:
                    print 'eta<=0'
                    continue

                # step 7: update the second alpha: j
                alphas[j] += label_mat[j] * (error_i-error_j)/eta
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print 'j not moving enough'
                    continue

                # step 8: update the first alpha: i
                # aplha[1] = alpha[i]_old + y1*y2*(alpha[2]_old-alpha[2])
                alphas[i] += label_mat[j]*label_mat[i] * (alpha_j_old-alpha_i_old)