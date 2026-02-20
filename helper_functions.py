import numpy as np
import time

def lu_factorization(A):
    """
    Performs LU factorization with partial pivoting.
    Returns permutation matrix P, lower triangular matrix L, and upper triangular matrix U.
    """
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n):

        pivot_index = np.argmax(np.abs(U[k:n, k])) + k

        if U[pivot_index, k] == 0:
            raise ValueError("Matrix is singular.")

        if pivot_index != k:

            U[[k, pivot_index]] = U[[pivot_index, k]]
            P[[k, pivot_index]] = P[[pivot_index, k]]

            if k > 0:
                L[[k, pivot_index], :k] = L[[pivot_index, k], :k]

        for i in range(k+1, n):

            multiplier = U[i, k] / U[k, k]

            L[i, k] = multiplier

            U[i, k:] = U[i, k:] - multiplier * U[k, k:]

    return P, L, U


def forward_substitution(L, b):
    """
    Solves Ly = b using forward substitution.
    """
    n = len(b)
    y = np.zeros(n)

    for i in range(n):

        sum_value = 0

        for j in range(i):
            sum_value += L[i, j] * y[j]

        y[i] = b[i] - sum_value

    return y


def backward_substitution(U, y):
    """
    Solves Ux = y using backward substitution.
    """
    n = len(y)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):

        sum_value = 0

        for j in range(i+1, n):
            sum_value += U[i, j] * x[j]

        x[i] = (y[i] - sum_value) / U[i, i]

    return x


def lu_solve(P, L, U, b):
    """
    Solves Ax = b using LU components.
    """
    Pb = np.dot(P, b)

    y = forward_substitution(L, Pb)

    x = backward_substitution(U, y)

    return x


def compute_backward_error(P, A, L, U):
    """
    Computes backward error.
    """
    return np.linalg.norm(np.dot(P, A) - np.dot(L, U)) / np.linalg.norm(A)


def compute_residual(A, x, b):
    """
    Computes residual.
    """
    return np.linalg.norm(np.dot(A, x) - b) / np.linalg.norm(b)


def generate_system(n):
    """
    Generates random system Ax=b.
    """
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    return A, b


def time_factorization(A):
    """
    Times LU factorization.
    """
    start = time.perf_counter()
    P, L, U = lu_factorization(A)
    end = time.perf_counter()

    return P, L, U, end - start


def time_solution(P, L, U, b):
    """
    Times solution phase.
    """
    start = time.perf_counter()
    x = lu_solve(P, L, U, b)
    end = time.perf_counter()

    return x, end - start
