from builtins import super
import numpy as np
import numbers
from .emregistration import EMRegistration


def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :,  :]
    diff = np.square(diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta**2))

def low_rank_eigen(G, num_eig):
    """
    Calculate num_eig eigenvectors and eigenvalues of gaussian matrix G.
    Enables lower dimensional solving.
    """
    S, Q = np.linalg.eigh(G)
    eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


class ConstrainedDeformableRegistration(EMRegistration):
    """
    Constrained deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.

    e_alpha: float (positive)
        Reliability of correspondence priors. Between 1e-8 (very reliable) and 1 (very unreliable)
    
    source_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the source array

    target_id: numpy.ndarray (int) 
        Indices for the points to be used as correspondences in the target array

    """

    def __init__(self, alpha=None, beta=None, e_alpha = None, source_id = None, target_id= None, low_rank=False, num_eig=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))

        if beta is not None and (not isinstance(beta, numbers.Number) or beta <= 0):
            raise ValueError(
                "Expected a positive value for the width of the coherent Gaussian kerenl. Instead got: {}".format(beta))
        
        if e_alpha is not None and (not isinstance(e_alpha, numbers.Number) or e_alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter e_alpha. Instead got: {}".format(e_alpha))
        
        if type(source_id) is not np.ndarray or source_id.ndim != 1:
            raise ValueError(
                "The source ids (source_id) must be a 1D numpy array of ints.")
        
        if type(target_id) is not np.ndarray or target_id.ndim != 1:
            raise ValueError(
                "The target ids (target_id) must be a 1D numpy array of ints.")

        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if beta is None else beta
        self.e_alpha = 1e-8 if e_alpha is None else e_alpha
        self.source_id = source_id
        self.target_id = target_id
        self.P_tilde = np.zeros((self.M, self.N))
        self.P_tilde[self.source_id, self.target_id] = 1
        self.P1_tilde = np.sum(self.P_tilde, axis=1)
        self.PX_tilde = np.dot(self.P_tilde, self.X)
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1./self.S)
            self.S = np.diag(self.S)
            self.E = 0.

    def update_transform(self):
        """
        Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.

        """
        if self.low_rank is False:
            A = np.dot(np.diag(self.P1), self.G) + \
                self.sigma2*(1/self.e_alpha)*np.dot(np.diag(self.P1_tilde), self.G) + \
                self.alpha * self.sigma2 * np.eye(self.M)
            B = self.PX - np.dot(np.diag(self.P1), self.Y) + self.sigma2*(1/self.e_alpha)*(self.PX_tilde - np.dot(np.diag(self.P1_tilde), self.Y)) 
            self.W = np.linalg.solve(A, B)

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1) + self.sigma2*(1/self.e_alpha)*np.diag(self.P1_tilde)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.dot(np.diag(self.P1), self.Y) + self.sigma2*(1/self.e_alpha)*(self.PX_tilde - np.dot(np.diag(self.P1_tilde), self.Y)) 

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
                np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        """
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + np.dot(G, self.W)
        else:
            if self.low_rank is False:
                self.TY = self.Y + np.dot(self.G, self.W)

            elif self.low_rank is True:
                self.TY = self.Y + np.matmul(self.Q, np.matmul(self.S, np.matmul(self.Q.T, self.W)))
                return


    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the deformable transformation.
        See the update rule for sigma2 in Eq. 23 of of https://arxiv.org/pdf/0905.2635.pdf.

        """
        qprev = self.sigma2

        # The original CPD paper does not explicitly calculate the objective functional.
        # This functional will include terms from both the negative log-likelihood and
        # the Gaussian kernel used for regularization.
        self.q = np.inf

        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X, self.X), axis=1))
        yPy = np.dot(np.transpose(self.P1),  np.sum(
            np.multiply(self.TY, self.TY), axis=1))
        trPXY = np.sum(np.multiply(self.TY, self.PX))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.
        self.diff = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.

        """
        return self.G, self.W
