# =================================================================================================
# @Author: Junbo Xin
# @Date: 2015/02/06
# @Description:  Ridge Regression
# =================================================================================================


from numpy import *
import regression


# giving training set: x_mat, y_mat, lamda(cannot use this word because it's keywords), return ws
def ridge_regression(x_mat, y_mat, lam=0.2):
    x = x_mat.T * x_mat   # (n*m) * (m*n) = n*n
    denom = x + eye(shape(x_mat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print 'This matrix is singular, cannot reverse'
        return
    ws = denom.I * x_mat.T * y_mat   # (n*n) * (n*m) * (m*1)
    return ws


def ridge_train(x_arr, y_arr):
    # normalize the data
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T

    y_mean = mean(y_mat, 0)
    y_mat = y_mat - y_mean

    x_mean = mean(x_mat, 0)
    x_var = mean(x_mat, 0)
    x_mat = (x_mat - x_mean)/x_var
    num_test = 30
    w_mat = zeros((num_test, shape(x_mat)[1]))  # 30*n
    for i in range(num_test):
        ws = ridge_regression(x_mat, y_mat, exp(i-10))
        w_mat[i, :] = ws.T    # each line is a 1*n vector
    return w_mat


def ridge_test():
    import matplotlib.pyplot as plt
    x_mat, y_mat = regression.load_data_set('abalone.txt')
    ridge_weights = ridge_train(x_mat, y_mat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridge_weights)
    plt.show()






