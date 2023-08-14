from builtins import super
import numpy as np
import numbers
from .emregistration import EMRegistration
from .utility import gaussian_kernel, low_rank_eigen

class DeformableRegistration(EMRegistration):
    """
    Deformable registration.

    Attributes
    ----------
    alpha: float (positive)
        Represents the trade-off between the goodness of maximum likelihood fit and regularization.

    beta: float(positive)
        Width of the Gaussian kernel.
    
    low_rank: bool
        Whether to use low rank approximation.
    
    num_eig: int
        Number of eigenvectors to use in lowrank calculation.
    """

    def __init__(self, alpha=2, beta=2, low_rank=False, num_eig=100,*args, **kwargs):
        super().__init__(*args, **kwargs)
        assert alpha > 0, "alpha must be positive"
        assert beta > 0, "beta must be positive"

        self.alpha = alpha
        self.beta = beta
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)
        self.low_rank = low_rank
        self.num_eig = num_eig
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = np.diag(1./self.S)
            self.S = np.diag(self.S)
            self.E = 0.

    def register(self, callback=lambda **kwargs: None):
        """
        Perform the EM registration.

        Attributes
        ----------
        callback: function
            A function that will be called after each iteration.
            Can be used to visualize the registration process.
        
        Returns
        -------
        self.TY: numpy array
            MxD array of transformed source points.
        
        registration_parameters:
            Returned params dependent on registration method used. 
        """
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance and self.diff<=self.diff_bound:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                          'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()
    
    def expectation(self):
        P = np.sum((self.X[None, :, :] - self.TY[:, None, :])**2, axis=2) # (M, N)
        c = 0
        
        if isinstance(self.sigma2, numbers.Number):
            P = np.exp(-P/(2*self.sigma2))
            c = (2*np.pi*self.sigma2)**(self.D/2)*self.w/(1. - self.w)*self.M/self.N
        else:
            P = np.exp(-P/(2*self.sigma2[:, None]))
            c = (2*np.pi*np.mean(self.sigma2))**(self.D/2)*self.w/(1. - self.w)*self.M/self.N

        den = np.sum(P, axis = 0, keepdims = True) # (1, N)
        den = np.clip(den, np.finfo(self.X.dtype).eps, None) + c

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)
    
    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        # self.update_variance()    # FIXME: Incorporate this step after debugging update_transform()

    def update_transform(self):
        """ Calculate a new estimate of the deformable transformation.
        See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf.
        """
        if self.low_rank is False:
            if isinstance(self.sigma2, numbers.Number):
                A = np.dot(np.diag(self.P1), self.G) + \
                    self.alpha * self.sigma2 * np.eye(self.M) 
                B = self.PX - (np.diag(self.P1) @ self.Y)
            else: # FIXME: The derivation of A is still wrong.
                # self.sigma2 = np.ones(self.M) * np.mean(self.sigma2)
                
                # A =  self.G @ np.diag(self.P1) + \
                    # self.alpha * np.diag(self.sigma2)    # Correct derivation, but cause large sigma2
                # B = self.PX - (np.diag(self.P1) @ self.Y)
                # A = np.diag(self.P1) @ self.G + \
                    # self.alpha * np.diag(self.sigma2)       # Good behaving sigma2, but derivation is wrong

                ''' No d(P1) inv multiplied, straight up'''
                dP1_inv = np.linalg.pinv(np.diag(self.P1))
                A = self.G + self.alpha * (dP1_inv @ np.diag(self.sigma2))
                B = dP1_inv @ self.PX - self.Y
            
            # self.W = np.linalg.solve(A, B)
            self.W = np.linalg.pinv(A) @ B
            

        elif self.low_rank is True:
            # Matlab code equivalent can be found here:
            # https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift
            dP = np.diag(self.P1)
            dPQ = np.matmul(dP, self.Q)
            F = self.PX - np.matmul(dP, self.Y)

            self.W = 1 / (self.alpha * self.sigma2) * (F - np.matmul(dPQ, (
                np.linalg.solve((self.alpha * self.sigma2 * self.inv_S + np.matmul(self.Q.T, dPQ)),
                                (np.matmul(self.Q.T, F))))))
            QtW = np.matmul(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2 * np.trace(np.matmul(QtW.T, np.matmul(self.S, QtW)))

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the deformable transformation.

        Attributes
        ----------
        Y: numpy array, optional
            Array of points to transform - use to predict on new set of points.
            Best for predicting on new points not used to run initial registration.
                If None, self.Y used.
        
        Returns
        -------
        If Y is None, returns None.
        Otherwise, returns the transformed Y.
                

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

        # Assume all \sigma_m^2 are the same
        if isinstance(self.sigma2, numbers.Number): 
            # The original CPD paper does not explicitly calculate the objective functional.
            # This functional will include terms from both the negative log-likelihood and
            # the Gaussian kernel used for regularization.
            self.q = np.inf

            xPx = np.dot(np.transpose(self.Pt1), np.sum(
                np.multiply(self.X, self.X), axis=1))
            yPy = np.dot(np.transpose(self.P1),  np.sum(
                np.multiply(self.TY, self.TY), axis=1))
            trPXY = np.sum(np.multiply(self.TY, self.PX))

            sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)
            if sigma2 <= 0: sigma2 = self.tolerance / 10
        
        # Assume each \sigma_m^2 is different
        else:   
            diff2 = np.linalg.norm(self.TY[:,None,:] - self.X, axis=-1, ord=2)**2  # (M,1,3) - (N,3) -> (M,N)
            weighted_diff2 = self.P * diff2             # (M,N)
            denom = np.sum(self.P, axis=1)[:,None]      # (M,1)
            sigma2 = np.sum(weighted_diff2 / denom, axis=1) / self.D

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.    
        self.diff = np.mean(np.abs(sigma2 - qprev))
        if self.diff <= self.diff_bound: 
            self.sigma2 = sigma2
            self.updated_variance = True

    def get_registration_parameters(self):
        """
        Return the current estimate of the deformable transformation parameters.


        Returns
        -------
        self.G: numpy array
            Gaussian kernel matrix.

        self.W: numpy array
            Deformable transformation matrix.
        """
        return self.G, self.W
