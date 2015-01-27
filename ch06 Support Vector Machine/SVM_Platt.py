# ======================================================================================================
# @Author: Junbo Xin
# @Date: 2015/01/27
# @Description: Platt SMO algorithm to implement SVM
# ======================================================================================================

from basic import *
from numpy import *
import time


# define the structure of svm for storing data
# add kernel to adjust non-linear case
class SVMStuct:
    def __init__(self, data_input, label_input, C, toler, kernel):
        self.train_x = data_input
        self.train_y = label_input
        self.C = C
        self.tol = toler
        self.m = shape(data_input)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0.0
        self.error_cache = mat(zeros((self.m, 2)))  # 1st col: valid or not,  2rd: error value
        self.kernel = mat(zeros((self.m, self.m)))  # k(xi,xj) : i,j =1,2,...m, so kernel is m*m matrix
        # for each row in training set, update kernel's col
        for i in range(self.m):
            self.kernel[:, i] = kernel_calc(self.train_x, self.train_x[i, :], kernel)


# calculate the k-th sample's error
def calc_k_error(svm, k):
    predict_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernel[:, k] + svm.b)
    # predict_k = float(multiply(svm.alphas, svm.train_y).T*(svm.train_x*svm.train_x[k, :].T)) + svm.b
    error_k = predict_k - float(svm.train_y[k])
    return error_k


# giving first alpha: i and its error, select the second alpha: j
def select_j(svm, i, error_i):
    j = -1
    max_error = 0
    error_j = 0.0
    svm.error_cache[i] = [1, error_i]
    # select only the valid error that has been update(non-zero)
    valid_error_list = nonzero(svm.error_cache[:, 0].A)[0]
    # if there are more than one alpha that is nonzero, then select the one maximize Ei-Ej
    if len(valid_error_list) > 1:
        for k in valid_error_list:
            if k == i:
                continue
            error_k = calc_k_error(svm, k)
            delta_error = abs(error_k - error_i)
            if delta_error > max_error:
                max_error = delta_error
                error_j = error_k
                j = k
    # else randomly select one alpha j as the second alpha
    else:
        j = select_rand(i, svm.m)
        error_j = calc_k_error(svm, j)
    return j, error_j


def update_error_k(svm, k):
    error_k = calc_k_error(svm, k)
    svm.error_cache[k] = [1, error_k]


# inner loop of svm, return wheter the current alpha pair has changed(1 for changed)
def inner_loop(svm, i):
    error_i = calc_k_error(svm, i)

    # check and pick the alpha that violates the KKT condition
    # satisfy KKT condition:
    # (1) y(i)*g(xi) >= 1  <====> alpha == 0    (outside the boundary)
    # (2) y(i)*g(xi) == 1  <====> 0<alpha<C     (on the boundary)
    # (3) y(i)*g(xi) <= 1  <====> alpha == C    (between the boundary)
    # ==============================================================================================
    # violate KKT condition
    # y(i) * error_i = y(i)*g(xi)- y(i)^2 = y(i)*f(xi) - 1
    # (1) if y(i) * error_i < 0, so y(i) * f(xi) < 1, if alpha < C, violate (alpha == C)
    # (2) if y(i) * error_i > 0, so y(i) * f(xi) > 1, if alpha > 0, violate (alpha == 0)
    # (3) if y(i) * error_i = 0, so y(i) * f(xi) = 1, it is on boundary, no need to optimize
    if ((svm.train_y[i]*error_i < -svm.tol) and (svm.alphas[i] < svm.C)) or \
       ((svm.train_y[i]*error_i > svm.tol) and (svm.alphas[i] > 0)):

        # ==================================step 1: select alpha j==================================
        j, error_j = select_j(svm, i, error_i)
        alpha_i_old = svm.alphas[i].copy()
        alpha_j_old = svm.alphas[j].copy()

        # ================step 2: calulate the boundary L and H for the second alpha: j==============
        if svm.train_y[i] != svm.train_y[j]:
            L = max(0, svm.alphas[j]-svm.alphas[i])
            H = min(svm.C, svm.C+svm.alphas[j]-svm.alphas[i])
        else:
            L = max(0, svm.alphas[j]+svm.alphas[i]-svm.C)
            H = min(svm.C, svm.alphas[j]+svm.alphas[i])
        if L == H:
            print 'L == H'
            return 0

        # ========================step 3: calculate eta: distance between i and j=====================
        # eta = svm.train_x[i, :]*svm.train_x[i, :].T + svm.train_x[j, :]*svm.train_x[j, :].T \
        #      - 2.0*svm.train_x[i, :]*svm.train_x[j, :].T
        eta = svm.kernel[i, i] + svm.kernel[j, j] - 2.0 * svm.kernel[i, j]
        if eta <= 0:
            print 'eta<=0'
            return 0

        # ================================= step 4: update alpha j ===================================
        svm.alphas[j] += svm.train_y[j]*(error_i-error_j)/eta
        svm.alphas[j] = adjust_alpha(svm.alphas[j], H, L)
        update_error_k(svm, j)

        # ========================step 5: if alpha j change little, return 0 ==========================
        if abs(svm.alphas[j] - alpha_j_old) < 0.00001:
            print 'j not moving enough'
            return 0

        # ========================step 6: update alpha i after optimized alpha j =======================
        svm.alphas[i] += svm.train_y[j]*svm.train_y[i]*(alpha_j_old-svm.alphas[j])
        update_error_k(svm, i)

        # ================================step 7: update the bias: b ===================================
        b1 = svm.b - error_i - svm.train_y[i]*(svm.alphas[i]-alpha_i_old)*svm.kernel[i, i] - \
                                svm.train_y[j]*(svm.alphas[j]-alpha_j_old)*svm.kernel[i, j]
        b2 = svm.b - error_j - svm.train_y[i]*(svm.alphas[i]-alpha_i_old)*svm.kernel[i, j] - \
                                svm.train_y[j]*(svm.alphas[j]-alpha_j_old)*svm.kernel[j, j]
        '''
        b1 = svm.b - error_i - svm.train_y[i]*(svm.alphas[i]-alpha_i_old)*\
             svm.train_x[i, :]*svm.train_x[i, :].T - svm.train_y[j]*\
             (svm.alphas[j]-alpha_j_old)*svm.train_x[i, :]*svm.train_x[j, :].T
        b2 = svm.b - error_j - svm.train_y[i]*(svm.alphas[i]-alpha_i_old)*\
             svm.train_x[i, :]*svm.train_x[j, :].T - svm.train_y[j]*\
             (svm.alphas[j]-alpha_j_old)*svm.train_x[j, :]*svm.train_x[j, :].T
        '''
        if (0 < svm.alphas[i]) and (svm.alphas[i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[j]) and (svm.alphas[j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1+b2)/2.0
        return 1

    else:
        return 0   # if alpha i did not violate KKT condition, then return 0


# improved platt smo algorithm
def smo_platt(data_input, label_input, C, toler, max_iter, kernel=('lin', 0)):
    start_time = time.time()
    svm = SVMStuct(mat(data_input), mat(label_input).transpose(), C, toler, kernel)

    # start training
    iter = 0
    entire_set = True
    alpha_pair_changed = 0
    # Iteration termination condition:
    # (1): reach max_iter:  iter == max_iter
    # (2): no alphas changed after going through all samples
    #      in other words, all alpha(samples) satisfy KKT conditon
    while (iter < max_iter) and ((alpha_pair_changed > 0) or (entire_set)):
        alpha_pair_changed = 0

        # update alphas over all training examples
        if entire_set:
            for i in range(svm.m):
                alpha_pair_changed += inner_loop(svm, i)
            print 'full set, iter: %d  i:%d, pairs changed %d' % (iter, i, alpha_pair_changed)
            iter += 1

        # update alphas over examples where alphas is not 0 and C (not on boundary)
        else:
            non_bounds = nonzero((svm.alphas.A > 0) * (svm.alphas.A < C))[0]
            for i in non_bounds:
                alpha_pair_changed += inner_loop(svm, i)
                print 'non bound, iter: %d  i:%d, pairs changed:%d' % (iter, i, alpha_pair_changed)
            iter += 1

        # alternate loop over all examples and non-boundary examples
        if entire_set:
            entire_set = False
        elif alpha_pair_changed:
            entire_set = True
        print 'iteration number:%d' % iter

    end_time = time.time()
    print 'Total Training time is: %fs' % (end_time - start_time)
    return svm.b, svm.alphas


# Giving alphas and the training data, calc w = sum(alphas[i]*y[i]*x[i])
def calc_weights(alphas, data_input, label_input):
    train_x = mat(data_input)
    train_y = mat(label_input).transpose()
    m, n = shape(train_x)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i]*train_y[i], train_x[i, :].T)
    return w


# @Input: train_x: m*n,  test_x: 1*n
# @Output: K: m*1
# @Description: for each row i of train_x(i=1,2,...,m), calc K[i] = Kernel(train_x[i], test[i])
# @rbf:   K(x, y) = -exp(||x-y||2^2)/(2*sigma^2)
def kernel_calc(train_x, test_x, kernel):
    m, n = shape(train_x)
    K = mat(zeros((m, 1)))
    if kernel[0] == 'lin':
        K = train_x * test_x.T
    elif kernel[0] == 'rbf':
        for j in range(m):
            delta = train_x[j, :] - test_x   # 1*n vector
            K[j] = delta * delta.T
        K = exp(K/-1*kernel[1]**2)   # kernel[1] is sigma
    else:
        raise NameError('Wrong kernel format!')
    return K   # K is a m*1 vector


# test rbf kernel svm
def test_rbf(k1=2.3):
    # step 1: obtain alphas and b, then get the decision boundary: y = sum(alpha[i]*y[i]*x[i]) * x + b
    data_arr, label_arr = load_data_set('testSetRBF.txt')
    b, alphas = smo_platt(data_arr, label_arr, 200, 0.0001, 10000, ('rbf', k1))
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    svm_index = nonzero(alphas.A > 0)[0]   # only non-zero alphas are support vector
    support_vec = data_mat[svm_index]
    label_sv = label_mat[svm_index]
    print 'there are %d support vectors' % shape(support_vec)[0]

    # step 2: calculate the training set error
    # y = w'x+b = sum(alphas[i] * y[i] * K(x, x[i])) + b
    m, n = shape(data_mat)
    error_count = 0
    for i in range(m):
        # only calc the support vector(non-zeros alphas)
        kernel_val = kernel_calc(support_vec, data_mat[i, :], ('rbf', k1))
        predict_y = kernel_val.T * multiply(label_sv, alphas[svm_index]) + b
        if sign(predict_y) != sign(label_arr[i]):
            error_count += 1
    print 'the training error rate is: %f' % (float(error_count)/m)

    # step 3: calculate the test set error
    test_arr, label_test_arr = load_data_set('testSetRBF2.txt')
    error_count = 0
    test_x = mat(test_arr)
    test_y = mat(label_test_arr).transpose()
    m, n = shape(test_x)
    for i in range(m):
        kernel_val = kernel_calc(support_vec, test_x[i, :], ('rbf', k1))
        predict_y = kernel_val.T* multiply(label_sv, alphas[svm_index]) + b
        if sign(predict_y) != sign(test_y[i]):
            error_count += 1
    print 'the test error rate is: %f' % (float(error_count)/m)



