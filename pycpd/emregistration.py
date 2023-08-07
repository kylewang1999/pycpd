from __future__ import division
import numpy as np, numbers, torch
from warnings import warn

def initialize_sigma2(X, Y, use_torch=False):
    """
    Initialize the variance (sigma2).

    Attributes
    ----------
    X: numpy array
        NxD array of points for target.
    
    Y: numpy array
        MxD array of points for source.
    
    Returns
    -------
    sigma2: float
        Initial variance.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    diff = X[None, :, :] - Y[:, None, :]
    err = diff ** 2
    
    if not use_torch:
        return np.sum(err) / (D*M*N)
    else:
        return torch.sum(err) / (D*M*N)

def lowrankQS(G, beta, num_eig, eig_fgt=False):
    """
    Calculate eigenvectors and eigenvalues of gaussian matrix G.
    
    !!!
    This function is a placeholder for implementing the fast
    gauss transform. It is not yet implemented.
    !!!

    Attributes
    ----------
    G: numpy array
        Gaussian kernel matrix.
    
    beta: float
        Width of the Gaussian kernel.
    
    num_eig: int
        Number of eigenvectors to use in lowrank calculation of G
    
    eig_fgt: bool
        If True, use fast gauss transform method to speed up. 
    """

    # if we do not use FGT we construct affinity matrix G and find the
    # first eigenvectors/values directly

    if eig_fgt is False:
        S, Q = np.linalg.eigh(G)
        eig_indices = list(np.argsort(np.abs(S))[::-1][:num_eig])
        Q = Q[:, eig_indices]  # eigenvectors
        S = S[eig_indices]  # eigenvalues.

        return Q, S

    elif eig_fgt is True:
        raise Exception('Fast Gauss Transform Not Implemented!')

class EMRegistration(object):
    """
    Expectation maximization point cloud registration.

    Attributes
    ----------
    X: numpy array
        NxD array of target points.

    Y: numpy array
        MxD array of source points.

    TY: numpy array
        MxD array of transformed source points.

    sigma2: float (positive)
        Initial variance of the Gaussian mixture model.

    N: int
        Number of target points.

    M: int
        Number of source points.

    D: int
        Dimensionality of source and target points

    iteration: int
        The current iteration throughout registration.

    max_iterations: int
        Registration will terminate once the algorithm has taken this
        many iterations.

    tolerance: float (positive)
        Registration will terminate once the difference between
        consecutive objective function values falls within this tolerance.

    w: float (between 0 and 1)
        Contribution of the uniform distribution to account for outliers.
        Valid values span 0 (inclusive) and 1 (exclusive).

    q: float
        The objective function value that represents the misalignment between source
        and target point clouds.

    diff: float (positive)
        The absolute difference between the current and previous objective function values.

    P: numpy array
        MxN array of probabilities.
        P[m, n] represents the probability that the m-th source point
        corresponds to the n-th target point.

    Pt1: numpy array
        Nx1 column array.
        Multiplication result between the transpose of P and a column vector of all 1s.

    P1: numpy array
        Mx1 column array.
        Multiplication result between P and a column vector of all 1s.

    Np: float (positive)
        The sum of all elements in P.

    """

    def init_sanity_check(self, X,Y,sigma2,max_iterations, tolerance, w):
        assert len(X.shape) == 2, f"The target point cloud (X) must be at a 2D numpy array, but got {X.shape}"
        assert len(Y.shape) == 2, f"The source point cloud (Y) must be a 2D numpy array, but got {Y.shape}"
        assert X.shape[1] == Y.shape[1], f"Both point clouds need to have the same number of dimensions, but got {X.shape}, {Y.shape}"
        
        if sigma2 is not None and \
            (not isinstance(sigma2, numbers.Number) or sigma2 <= 0) and \
            (not isinstance(sigma2, np.ndarray) and not isinstance(sigma2, torch.Tensor)):
            raise ValueError(
                "Expected a positive value for sigma2 instead got: {}".format(sigma2))
        
        if max_iterations is not None and (not isinstance(max_iterations, numbers.Number) or max_iterations < 0):
            raise ValueError(
                "Expected a positive integer for max_iterations instead got: {}".format(max_iterations))
        elif isinstance(max_iterations, numbers.Number) and not isinstance(max_iterations, int):
            warn("Received a non-integer value for max_iterations: {}. Casting to integer.".format(max_iterations))
            max_iterations = int(max_iterations)
        
        if tolerance is not None and (not isinstance(tolerance, numbers.Number) or tolerance < 0):
            raise ValueError(
                "Expected a positive float for tolerance instead got: {}".format(tolerance))
        
        if w is not None and (not isinstance(w, numbers.Number) or w < 0 or w >= 1):
            raise ValueError(
                "Expected a value between 0 (inclusive) and 1 (exclusive) for w instead got: {}".format(w))

    def __init__(self, X, Y, sigma2=None, max_iterations=None, tolerance=None, w=None, \
                use_torch=False,*args, **kwargs):
        self.init_sanity_check(X, Y, sigma2, max_iterations, tolerance, w)

        self.X = X; self.Y = Y; self.TY = Y
        self.sigma2 = initialize_sigma2(X, Y) if sigma2 is None else sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = 0.001 if tolerance is None else tolerance
        self.w = 0.0 if w is None else w
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.use_torch = use_torch
        
        self.Np = 0
        self.diff = torch.inf if self.use_torch else np.inf 
        self.q = torch.inf if self.use_torch else np.inf
        self.P = torch.zeros((self.M, self.N)) if self.use_torch else np.zeros((self.M, self.N))
        self.Pt1 = torch.zeros((self.N, )) if self.use_torch else np.zeros((self.N, ))
        self.P1 = torch.zeros((self.M, )) if self.use_torch else np.zeros((self.M, ))
        self.PX = torch.zeros((self.M, self.D)) if self.use_torch else np.zeros((self.M, self.D))
            
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
        while self.iteration < self.max_iterations and np.mean(self.diff) > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration,
                        'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_registration_parameters()

    def iterate(self):
        """
        Perform one iteration of the EM algorithm.
        """
        if not self.use_torch:
            self.expectation(); self.maximization()
        self.iteration += 1

    def expectation(self):
        """
        Compute the expectation step of the EM algorithm.

        Kaiyuan's Modification: Generalize the solution such that each \sigma_m^2 can be different.
        """
        if not self.use_torch:
            P = np.sum((self.X[None, :, :] - self.TY[:, None, :])**2, axis=2) # (M, N)

            if isinstance(self.sigma2, numbers.Number): #  Assume all \sigma_m^2 are the same 
                P = np.exp(-P/(2*self.sigma2))
                c = (2*np.pi*self.sigma2)**(self.D/2)*self.w/(1. - self.w)*self.M/self.N

            else:                                       # Assume each \sigma_m^2 can be different
                P = np.exp(-P/(2*self.sigma2[:,None]))
                c = (2*np.pi*np.mean(self.sigma2))**(self.D/2)*self.w/(1. - self.w)*self.M/self.N

            den = np.sum(P, axis = 0, keepdims = True) # (1,N)
            den = np.clip(den, np.finfo(self.X.dtype).eps, None) + c

            self.P = P / den
            self.Pt1 = np.sum(self.P, axis=0)
            self.P1 = np.sum(self.P, axis=1)
            self.Np = np.sum(self.P1)
            self.PX = self.P @ self.X
        
        else:
            P = torch.sum((self.X[None, :, :] - self.TY[:, None, :])**2, axis=2) # (M, N)

            if isinstance(self.sigma2, numbers.Number): #  Assume all \sigma_m^2 are the same 
                P = torch.exp(-P/(2*self.sigma2))
                c = (2*torch.pi*self.sigma2)**(self.D/2)*self.w/(1. - self.w)*self.M/self.N

            else:                                       # Assume \sigma_m^2 can be different
                P = torch.exp(-P/(2*self.sigma2[:,None]))
                c = (2*torch.pi*torch.mean(self.sigma2))**(self.D/2)*self.w/(1. - self.w)*self.M/self.N

            den = torch.sum(P, axis = 0, keepdims = True) # (1,N)
            den = torch.clip(den, torch.finfo(self.X.dtype).eps, None) + c

            self.P = P / den
            self.Pt1 = torch.sum(self.P, axis=0)
            self.P1 = torch.sum(self.P, axis=1)
            self.Np = torch.sum(self.P1)
            self.PX = self.P @ self.X

    def maximization(self):
        """
        Compute the maximization step of the EM algorithm.
        """
        if not self.use_torch:
            self.update_transform()
            self.transform_point_cloud()
            self.update_variance()

        else:
            try: self.torch_e2e_optimize()
            except: raise ValueError("Torch optimization failed.")


    ''' Function place holders to be inherited by child classes '''

    def get_registration_parameters(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Registration parameters should be defined in child classes.")

    def update_transform(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating transform parameters should be defined in child classes.")

    def transform_point_cloud(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the source point cloud should be defined in child classes.")

    def update_variance(self):
        """
        Placeholder for child classes.
        """
        raise NotImplementedError(
            "Updating the Gaussian variance for the mixture model should be defined in child classes.")
