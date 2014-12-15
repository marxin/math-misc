#!/usr/bin/env python

import numpy as np
from operator import *
from math import *
import matplotlib.pyplot as plt

def GEPP(A, b, doPricing = True):
    '''
    Gaussian elimination with partial pivoting.
    
    input: A is an n x n numpy matrix
           b is an n x 1 numpy array
    output: x is the solution of Ax=b 
            with the entries permuted in 
            accordance with the pivoting 
            done by the algorithm
    post-condition: A and b have been modified.
    '''
    n = len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between"+
                         "A & b.", b.size, n)
    # k represents the current pivot row. Since GE traverses the matrix in the 
    # upper right triangle, we also use k for indicating the k-th diagonal 
    # column index.
    
    # Elimination
    for k in range(n-1):
        if doPricing:
            # Pivot
            maxindex = abs(A[k:,k]).argmax() + k
            if A[maxindex, k] == 0:
                raise ValueError("Matrix is singular.")
            # Swap
            if maxindex != k:
                A[[k,maxindex]] = A[[maxindex, k]]
                b[[k,maxindex]] = b[[maxindex, k]]
        else:
            if A[k, k] == 0:
                raise ValueError("Pivot element is zero. Try setting doPricing to True.")
        #Eliminate
        for row in range(k+1, n):
            multiplier = A[row,k]/A[k,k]
            A[row, k:] = A[row, k:] - multiplier*A[k, k:]
            b[row] = b[row] - multiplier*b[k]
    # Back Substitution
    x = np.zeros(n)
    for k in range(n-1, -1, -1):
        x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
    return x

# A = np.array([[1.,-1.,1.,-1.],[1.,0.,0.,0.],[1.,1.,1.,1.],[1.,2.,4.,8.]])
# b =  np.array([[14.],[4.],[2.],[2.]])

def dot(vector1, vector2):
  r = sum(map(mul, vector1, vector2))
  return r

# x = range(0, 7)
# y = [0.1, 0.93, 0.67, 0.11, -0.37, -1.02, -0.2]
# f = [lambda x: 1, lambda x: sin(x)]

x = [0, 1, 2]
y = [2.1, 3.6, 4.7]
f = [lambda x: x, lambda x: 1]

# x = range(-1, 3)
# fx = lambda x: x**2
# y = list(map(fx, x))
# f = [lambda x: 1, lambda x: exp(x), lambda x: exp(-x)]

phis = map(lambda b: list(map(lambda a: b(a), x)), f)
l = len(phis)

A = np.zeros((l, l))

for i in range(l):
  for j in range(l):
    A[i][j] = dot(phis[i], phis[j])

b = np.zeros((l))


for i in range(l):
  b[i] = dot(phis[i], y)


print(A)
print(b)

# print GEPP(np.copy(A), np.copy(b), doPricing = False)
solution = GEPP(A,b)

print('Approximation solution')
print(solution)

approx = np.zeros((len(x)))
getval = lambda value: sum(map(lambda a: a[1] * a[0](value), zip(f, solution)))

for i in range(len(x)):
  # y(x) = a * 1 + b * sin(x)
  approx[i] = getval(x[i])

print('Approximation differences')
print(y - approx)

print('Approximation diff squares')
sq = list(map(lambda x: x**2, y - approx))
print('SUM squares: %f' % sum(sq))

X = np.linspace(x[0], x[-1], 256, endpoint=True)
F = map(lambda v: getval(v), X)

# X2 = np.linspace(-10, 3, 100, endpoint=True)
# F2 = map(lambda v: exp(v) - v - 5, X2)

plt.plot(X, F)
plt.plot(x, y, 'ro')

plt.savefig('/tmp/fig.pdf')

# Hermit polynom solver
print('Hermit interpolation polynom solver')

x = [-0.1, 0, 0.1]
xx = reduce(add, map(lambda a: [a, a], x))
# y = [-1, 0, 1]
# y = [2, 0, 1]
y = [-1.1, 0.2, 0.9]

# y2 = [0, 0, 0]
#y2 = [1, 1, 3]
# y2 = list(map(lambda a: 1 + cos(a), x))
y2 = [13, 8, 10]

l = 2 * len(x)
V = np.zeros((l, l))

for i in range(l):
  V[i][0] = y[i / 2]

for i in range(1, l):
  for j in range(l - i):
    a = V[j][i-1] - V[j+1][i-1]
    b = xx[j] - xx[i + j]
    r = 0
    if a == 0 and b == 0:
      r = y2[j / 2]
    else:
      r = a / b
    V[j][i] = r

np.set_printoptions(formatter={'float': '{: 3.2f}'.format})
print(V)

# Cholesky decomposition
print("Cholesky decomposition")
A = np.array([[2, 0, -1], [0, 3, 2], [-1, 2, 4]])
d = np.linalg.cholesky(A)

print(d)
print(np.dot(d, d.T.conj()))

# Newton for 2 nonlinear equations
# values = [[0, 0]]
values = [[1.7, -4]]

print("Gaus 2D")
# functions = [lambda x, y: x - exp(y) + x**3, lambda x, y: y - 2*exp(x) + y**3]
functions = [lambda x, y: x-1/y-2, lambda x, y: (x-2)**2+((y+2)**2)/4-1]
l = len(functions)
# Jacobi = np.array([[lambda x, y: 1 + 3 * x**2, lambda x, y: -exp(y)], [lambda x, y: -2*exp(x), lambda x, y: 1 + 3 * y**2]])
Jacobi = np.array([[lambda x, y: 1, lambda x, y: y**-2], [lambda x, y: 2*x-4, lambda x, y: y/2+1]])

for i in range(5):
  print('Iteration: %d' % i)
  current = values[-1]
  der = np.reshape(list(map(lambda x: x(current[0], current[1]), Jacobi.flatten())), (l, l))
  right = np.array(map(lambda f: -f(current[0], current[1]), functions))

  solution = GEPP(der.copy(), right.copy())
  print('New difference')
  print(solution)
  next_value = solution + current
  print('New value:')
  print(next_value)
  values.append(next_value)

F = lambda x: x**2 - atan(2*x)
FD = lambda x: 2*x-2/(1+4*x**2)
values = [1]

print('=== Newton ===')
for i in range(10):
  prev = values[-1]
  new = prev - F(prev) / FD(prev)
  values.append(new)
  print('New value: %f, difference: %f' % (new, F(new)))


print('Network method')

A = np.array([[8, -2, 0, 0], [-6, 8, -2, 0], [0, -6, 8, -2], [0, 0, -8, 16]])
b = np.array([7, 1, 1, 1])

print(A)
print(b)
solution = GEPP(A, b)
print(solution)
