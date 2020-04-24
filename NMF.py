import numpy as np
from scipy.sparse.linalg import svds
EPSILON = np.finfo(np.float32).eps


def trace_dot(X, Y):
    return np.dot(X.ravel(), Y.ravel())


class NMF:
    def __init__(self, n_components, max_iterations=200, tolerance=0.0001):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.W = None
        self.H = None
        self.rec_loss = None
        self.n_iterations = 0

    def sparse_l2_loss(self, V):
        norm_V = V.data @ V.data
        norm_WH = trace_dot((self.W.T @ self.W) @ self.H, self.H)
        cross_prod = trace_dot((V * self.H.T), self.W)
        res = (norm_V + norm_WH - 2. * cross_prod)/2.
        return res

    def __init_w_h(self, V, k):
        self.W, self.H = nndsvd(V,k)

    def decompose(self, V):
        self.__init_w_h(V, self.n_components)
        init_loss = self.sparse_l2_loss(V)
        self.rec_loss = init_loss
        while self.n_iterations < self.max_iterations:
            self.n_iterations += 1
            den = self.W.T @ self.W @ self.H
            den[den == 0] = EPSILON
            num = self.W.T @ V
            num /= den
            self.H *= num

            den = self.W @ self.H @ np.transpose(self.H)
            den[den == 0] = EPSILON
            num = V @ self.H.T
            num /= den
            self.W *= num

            if self.n_iterations % 10 == 0:
                new_loss = self.sparse_l2_loss(V)
                delta = self.rec_loss - new_loss
                self.rec_loss = new_loss
                if (delta / init_loss) < self.tolerance:
                    #print ('finished before max_iters')
                    break

        return self.W, self.H


def nndsvd(A, k):
    U, S, V = svds(A, k)
    W = np.zeros((A.shape[0], k))
    H = np.zeros((k, A.shape[1]))
    W[:, 0] = np.sqrt(S[0]) * U[:,0]
    H[1, :] = np.sqrt(S[0]) * V[0, :]
    for j in range(1, k):
        x = U[:, j]
        y = V[j, :]
        xp, xn = (x >= 0) * x, (x < 0) * (-x)
        yp, yn = (y >= 0) * y, (y < 0) * (-y)
        xpnrm, xnnrm = np.linalg.norm(xp), np.linalg.norm(xn)
        ypnrm, ynnrm = np.linalg.norm(yp), np.linalg.norm(yn)
        mp, mn = xpnrm * ypnrm, xnnrm * ynnrm
        if mp>mn:
            u, v, sigma = xp/xpnrm, yp/ypnrm, mp
        else:
            u, v, sigma = xn/xnnrm, yn/ynnrm, mn
        W[:, j] = np.sqrt(S[j] * sigma) * u
        H[j, :] = np.sqrt(S[j] * sigma) * v
    return W, H