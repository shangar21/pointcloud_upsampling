import numpy as np

class KalmanFilter(object):
    def __init__(self, A, B, G, H, Q, R, x_init, Sigma_init):
        self.x = x_init
        self.Sigma = Sigma_init
        self.A = A
        self.B = B
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R

    def predict(self, u=None):
        if u is None:
            u = np.zeros((self.B.shape[1], ))

        self.x = self.A @ self.x + self.B @ u
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.G @ self.Q @ self.G.T

    def update(self, z):
        k = self.Sigma @ self.H.T @ np.linalg.inv(self.H @ self.Sigma @ self.H.T + self.R)
        self.x = self.x + k @ (z - self.H @ self.x)
        self.Sigma = (np.eye(self.Sigma.shape[0]) - k @ self.H) @ self.Sigma
