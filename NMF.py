# My version of the Non-negative Matrix Factorization that implements a weight mask on the nan_values

import numpy as np
from scipy.sparse.linalg import svds

EPSILON = np.finfo(np.float32).eps


def trace_dot(X, Y):
    return np.dot(X.ravel(), Y.ravel())


class NMF:
    def __init__(self, n_components, apply_nan_mask=False, nan_weight=0,
                 max_iterations=200, tolerance=0.0001):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.W = None
        self.H = None
        self.rec_loss = None
        self.n_iterations = 0
        self.nan_mask = apply_nan_mask
        self.nan_weight = nan_weight

    def l2_loss(self, V):
        if self.nan_mask:
            WH = self.W @ self.H
            res = np.sum(np.square(np.multiply(self.M, (V - WH)))) / 2.
        else:
            norm_V = V.data @ V.data
            norm_WH = trace_dot((self.W.T @ self.W) @ self.H, self.H)
            cross_prod = trace_dot((V * self.H.T), self.W)
            res = (norm_V + norm_WH - 2. * cross_prod) / 2.
        return res

    def __init_w_h(self, V, k):
        self.W, self.H = nndsvd(V, k)

    def decompose(self, V):
        self.__init_w_h(V, self.n_components)
        if self.nan_mask:
            self.M = (V != 0)
            self.M = self.M.astype(float).toarray()
            self.M[self.M == 0] = self.nan_weight
        init_loss = self.l2_loss(V)
        self.rec_loss = init_loss
        while self.n_iterations < self.max_iterations:
            self.n_iterations += 1
            # update of H
            WH = self.W @ self.H
            if self.nan_mask:
                WH = self.M * WH
            den = self.W.T @ WH
            den[den == 0] = EPSILON
            if self.nan_mask:
                num = self.W.T @ V.multiply(self.M)
            else:
                num = self.W.T @ V
            num /= den
            self.H *= num

            # update of W
            WH = self.W @ self.H
            if self.nan_mask:
                WH = self.M * WH
            den = WH @ self.H.T
            den[den == 0] = EPSILON
            if self.nan_mask:
                num = V.multiply(self.M) @ self.H.T
            else:
                num = V @ self.H.T
            num /= den
            self.W *= num

            if self.n_iterations % 10 == 0:
                new_loss = self.l2_loss(V)
                delta = self.rec_loss - new_loss
                self.rec_loss = new_loss
                if (delta / init_loss) < self.tolerance:
                    # print('finished before max_iters: ', self.n_iterations)
                    break

        return self.W, self.H


def nndsvd(A, k):
    U, S, V = svds(A, k)
    W = np.zeros((A.shape[0], k))
    H = np.zeros((k, A.shape[1]))
    W[:, 0] = np.sqrt(S[0]) * U[:, 0]
    H[1, :] = np.sqrt(S[0]) * V[0, :]
    for j in range(1, k):
        x = U[:, j]
        y = V[j, :]
        xp, xn = (x >= 0) * x, (x < 0) * (-x)
        yp, yn = (y >= 0) * y, (y < 0) * (-y)
        xpnrm, xnnrm = np.linalg.norm(xp), np.linalg.norm(xn)
        ypnrm, ynnrm = np.linalg.norm(yp), np.linalg.norm(yn)
        mp, mn = xpnrm * ypnrm, xnnrm * ynnrm
        if mp > mn:
            u, v, sigma = xp / xpnrm, yp / ypnrm, mp
        else:
            u, v, sigma = xn / xnnrm, yn / ynnrm, mn
        W[:, j] = np.sqrt(S[j] * sigma) * u
        H[j, :] = np.sqrt(S[j] * sigma) * v
    return W, H
