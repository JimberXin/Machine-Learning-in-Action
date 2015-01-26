from basic import *
from numpy import *


class SVMStuct:
    def __init__(self, data_input, label_input, C, toler):
        self.train_x = data_input
        self.train_y = label_input
        self.C = C
        self.toler = toler
        self.m = shape(data_input)[0]
        self.alphas = mat(zeros((self.numSamples, 1)))
        self.b = 0.0
        self.error_cache = mat(zeros((self.m, 2)))  # 1st col: valid or not,  2rd: error value


# calculate the k-th sample's error
def calc_k_error(svm, k):
    predict_k = float(multiply(svm.alphas, sum.train_y).T*(svm.train_x*svm.train[k, :].T)) + svm.b
    error_k = predict_k - float(svm.train_y[k])
    return error_k


def select_j(svm, i, error_i):
    j = -1
    max_error = 0
    error_j = 0.0
    svm.error_cache[i] = [1, error_i]
    # select only the valid error that has been update(non-zero)
    valid_error_list = nonzero(svm.error_cache[:, 0].A)[0]
    if len(valid_error_list) > 1:
        for k in valid_error_list:
            if k == i: continue
            error_k = calc_k_error(svm, k)
            delta_error = abs(error_k - error_i)
            if delta_error > max_error:
                max_error = delta_error
                error_j = error_k
                j = k
    else:
        j = select_rand(i, svm.m)
        error_j = calc_k_error(svm, j)
    return j, error_j


def update_error_k(svm, k):
    error_k = calc_k_error(svm, k)
    svm.error_cache[k] = [1, error_k]


def inner_loop(svm, i):
    error_i = calc_k_error(svm, i)
    if ((svm.train_y[i]*error_i < -svm.toler) and (svm.alphas[i] < svm.C)) or \
       ((svm.train_y[i]*error_i > svm.toler) and (svm.alphas[i] > 0)):
        j, error_j = select_j(svm, i, error_i)
        alpha_i_old = svm.alphas[i].copy()
        alpha_j_old = svm.alphas[j].copy()

        if svm.train_y[i] != svm.train_y[j]:
            L = max(0, svm.alphas[j]-svm.alphas[i])
            H = min(svm.C, svm.C+svm.alphas[j]-svm.alphas[i])
        else:
            L = max(0, svm.alphas[j]+svm.alphas[i]-svm.C)
            H = min(svm.C, svm.alphas[j]+svm.alphas[i])
        if L == H:
            print 'L == H'
            return 0
        eta = svm.train_x[i, :]*svm.train_x[i, :].T + svm.train_x[j, :]*svm.train_x[j, :].T \
              - 2.0*svm.train_x[i, :]*svm.train_x[j, :].T
        if eta <= 0:
            print 'eta<=0'
            return 0
        svm.alphas[j] += svm.train_y*(error_i-error_j)/eta
        svm.alphas[j] = adjust_alpha(svm.alphas[j], L, H)
        update_error_k(svm, j)
        if abs(svm.alphas[j] - alpha_j_old) < 0.00001:
            print 'j not moving enough'
            return 0
        svm.alphas[i] += svm.train_y[j]*svm.train_y[i]*(alpha_j_old-svm.alphas[j])
        update_error_k(svm, i)
        b1 = svm.b - error_i - svm.train_y[i]*(svm.alphas[i]-alpha_i_old)*\
             svm.train_x[i, :]*svm.train_x[i, :].T - svm.train_y[j]*\
             (svm.alphas[j]-alpha_j_old)*svm.train_x[i, :]*svm.train_x[j, :].T
        b2 = svm.b - error_j - svm.train_y[i]*(svm.alphas[i]-alpha_i_old)*\
             svm.train_x[i, :]*svm.train_x[j, :].T - svm.train_y[j]*\
             (svm.alphas[j]-alpha_j_old)*svm.train_x[j, :]*svm.train_x[j, :].T
        if (0 < svm.alphas[i]) and (svm.alphas[i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[j]) and (svm.alphas[j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1+b2)/2.0
        return 1

    else:
        return 0


