# Markov-Chain-Analysis
# This Python code helps to create a toy example for Markov Chain and analyses its n-step behavior and stationary distribution.
import numpy as np


n = 12


np.random.seed(7)


P = np.random.rand(n, n)         
P = P / P.sum(axis=1, keepdims=True)  



pi = np.random.rand(n)
pi = pi / pi.sum()   


k=8
def matrix_power(matrix, k):

    mat = np.array(matrix)

    return np.linalg.matrix_power(mat, k)
result = matrix_power(P, k)


k_th_step = np.matmul(pi, result)

print("Transition Matrix P:")
print(P)



print("Initial Distribution π:")
print(pi)

print("P^{} =".format(k))
print(result)

print("k_th_step")
print(k_th_step)


A_initial = P.T - np.eye(n)
b_initial = np.zeros(n)


row_of_ones = np.ones((1, n))
A_1 = np.vstack([A_initial, row_of_ones])
b_1 = np.append(b_initial, 1)


stationary_dist = np.linalg.lstsq(A_1, b_1 )[0]

print("Stationary Distribution ")
print(stationary_dist)

Verify = np.matmul(stationary_dist, P)
print("P.pi = ", Verify)
