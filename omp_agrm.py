import numpy as np
import math
from basic_import import *

Sparse_degree = [2, 3, 5, 7, 10, 20, 30, 50, 100, 300, 1000]


# OMP algorithm representation
def omp(diction_with_error, b):
    residual = b
    index_matrix = []
    index_matrix_whole = []
    index_set = []
    last_residual = 0
    L = math.floor(Sparse_degree[6])
    # L = math.floor(diction_with_error[0].shape[0]*3/4)

    # iterate the omp process
    cnt = 0
    cnt_repre = 0
    for i in range(L):
        c_k = np.fabs(np.dot(diction_with_error.T, residual))    # dot choose the kth index
        # print(c_k)
        k = np.where(c_k == np.max(c_k))[0][0]              # position of the largest projection
        while k in index_set:
            c_k[k] = 0
            k = np.where(c_k == np.max(c_k))[0][0]
        index_set.append(k)  # update index set
        index_matrix.append(diction_with_error.T[k].tolist())     # update index_matrix set
        # index_matrix_whole.append(diction_with_error.T[k])
        A_k = np.array(index_matrix).T                  # transform the index_matrix to numpy form
        x_k = np.linalg.pinv(A_k.T.dot(A_k)).dot(A_k.T).dot(b)      #least squares method
        residual = b - A_k.dot(x_k)                     # compute the residual
        if abs(np.linalg.norm(residual)-np.linalg.norm(last_residual)) < 1e-8:
            cnt += 1
            if cnt >= 10:
                break
        # print(np.linalg.norm(residual), "  ", i, "/", L)# show the residual
        last_residual = residual
        if i+1 >= diction_with_error[0].shape[0]:
            break

    A_k = np.array(index_matrix).T                      # final support-dictionary matrix
    x_k = np.linalg.pinv(A_k.T.dot(A_k)).dot(A_k.T).dot(b)       # final support-presentation vector(include x and error)
    # A_whole_k = np.array(index_matrix_whole).T
    # x_whole_k = np.linalg.inv(A_whole_k.T.dot(A_whole_k)).dot(A_whole_k.T).dot(b_whole)

    x_hat = []                                          # final representation vector
    for t in range(diction_with_error[0].shape[0]):
        x_hat.append(0)
    for t in range(len(x_k)):
        x_hat[index_set[t]] = x_k[t]                    # construct complete
    x = np.array(x_hat)
    return x


def x_select(x, diction_with_error, b, gender):

    #########################
    #       Method 1        #
    #########################
    delta_x = []                                        # delta_x[i] means the vector only
    for i in range(50):                                 # contains the parameters of the ith class
        delta_x_i = []
        for j in range(700):
            if i*14 <= j <= i*14+13:
                delta_x_i.append(x[j])
            else:
                delta_x_i.append(0)
        delta_x.append(np.array(delta_x_i))
    delta_x = np.array(delta_x)

    r_set = []                                          # calculate the residual of every delta_x[i]
    for delta_x_i in delta_x:                           # select the vector with least residual
        r = b - diction_with_error.dot(delta_x_i)
        r_set.append(np.linalg.norm(r))
    r_set = np.array(r_set)
    k = np.where(r_set == np.min(r_set))[0][0]
    result1 = k+1

    ##########################
    #       Method 2         #
    ##########################
    x_sorted_index = np.argsort(-x)
    x_index = []
    for i in range(51):
        x_index.append(0)
    for i in range(20):
        if x[x_sorted_index[i]] > 0:
            file_index_string = find_file(x_sorted_index[i], gender)
            file_index_num = int(file_index_string[3:5])
            x_index[file_index_num] += x[x_sorted_index[i]]
    x_index = np.array(x_index)
    FCR_num = np.argsort(-x_index)[0]
    result2 = FCR_num

    return result1, result2


if __name__ == "__main__":
    A = np.random.rand(300000, 4)
    x = np.random.rand(4, 1)
    y = A.dot(x)
    x_rec = omp(A, y)
    residual = np.linalg.norm(x_rec-x)
    print(x)
    print("------------")
    print(x_rec)
    print("------------")
    print(x-x_rec)
    print("------------")
    print(residual)