from __future__ import division
from math import log
import numbers, numpy as np
import torch, torch.nn as nn
from tqdm import tqdm

try:
    from pycpd.mesh_gen import MeshGenerator
    from pycpd.utility import *
except ImportError:
    from pycpd.pycpd.mesh_gen import MeshGenerator
    from pycpd.pycpd.utility import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_sigma2(X, Y, P=None):
    """ Compute the variance (sigma2).

    Inputs:
        - X: torch.tensor (N,D). Target points
        - Y: torch.tensor (M,D). Source gmm centroids
        - P: torch.tensor (M,N). Soft centroid assignment matrix
    
    Returns:
        - sigma2 (M,). Per-point covariances
    """
    (N, D) = X.shape
    (M, _) = Y.shape

    if isinstance(X, torch.Tensor):
        if P is None: P = torch.ones((M,N), device=DEVICE)
        diff2 = torch.norm(Y[:,None,:] - X, dim=-1, p=2)**2  # (M,1,3) - (N,3) -> (M,N,3) -> (M,N)
        weighted_diff2 = P * diff2      # (M,N)
        denom = P.sum(dim=-1)[:,None]  # (M,1)
        sigma2 = torch.sum(weighted_diff2 / denom, dim=-1) / D  # (M, N) -> (M,)
        return sigma2
    
    else:
        if P is None: P = np.ones((M,N), dtype=np.float64)
        diff2 = np.linalg.norm(Y[:,None,:] - X, axis=-1, ord=2)**2  # (M,1,3) - (N,3) -> (M,N,3) -> (M,N)
        weighted_diff2 = P * diff2      # (M,N)
        denom = np.sum(P, axis=-1)[:,None]  # (M,1)
        sigma2 = np.sum(weighted_diff2 / denom, axis=-1) / D  # (M, N) -> (M,)
        return sigma2

def np2torch(X, dtype=torch.float64): 
    return torch.as_tensor(X, dtype=dtype, device=DEVICE) if isinstance(X, np.ndarray) else X

def torch2np(X):  
    return X.cpu().detach().numpy().squeeze() if isinstance(X, torch.Tensor) else X

class DeformableRegistrationLoss(nn.Module):
    ''' Implements loss function for deformable registration '''

    def __init__(self, alpha=0, beta=2): 
        '''
            - alpha: (scalar).  Regularization strength of CPD term
            - beta: (scalar).   Regularization strength of Geodesic term
        '''
        super(DeformableRegistrationLoss, self).__init__()
        self.alpha = alpha; self.beta = beta
        self.losses = {}

    def forward(self, X, Y, sigma2, P, G, W, edges=None):
        '''
        Inputs:
            - X: (N,D=3).       Target point cloud 
            - Y: (M,D).         Source gmm centroids
            - sigma2: (M,).     Variance of each gmm centroid
            - P: (M,N).         Soft cluster assignments
            - G: (M,M).         Y after gaussian kernel
            - W: (M,D).         Deformable transformation  \delta Y  = G @ W

        '''
        (N,D) = X.shape; (M,_) = Y.shape

        log_term = torch.log(sigma2)[:,None]                    # (M,1)
        diff_term = torch.norm(Y[:,None,:] - X, dim=-1, p=2)**2 # (M,N)
        diff_term /= sigma2[:,None]                             # (M,N)
        
        gmm_loss = D * log_term + diff_term                     # (M,N)
        gmm_loss = torch.sum(P * gmm_loss)     

        cpd_loss = torch.trace(W.T @ G @ W)                      

        geo_loss = 0
        if edges is not None:
            geo_loss += torch.norm(Y[edges[...,0]] - Y[edges[...,1]], dim=-1, p=2).sum()
            
        loss = gmm_loss + self.alpha * cpd_loss + self.beta * geo_loss
        self.losses['gmm_loss'] = gmm_loss.item()
        self.losses['cpd_loss'] = cpd_loss.item()
        self.losses['geo_loss'] = geo_loss.item()
        
        return loss

class DeformableRegistrationTorch(object):

    def __init__(self, X, Y, P=None, sigma2=None, max_iterations=100, tolerance=1e-3, 
        alpha=2, beta=2, w=0.5, solver='em', optim_config=None, *args, **kwargs):

        assert solver in ['em', 'torch'], f'Solver {solver} not supported. Expect \'em\' for Expectation-Maximization or \'torch\' for PyTorch autograd'
        self.solver = solver
        self.optim_config = {
            'W': {'lr': 5e-5, 'epochs': 100},
        } if optim_config is None else optim_config
        self.device = DEVICE; 
        self.mesh_generator = MeshGenerator(Y, method='knn')

        self.X = np2torch(X); self.Y = np2torch(Y); self.TY = np2torch(Y)
        self.alpha = alpha; self.beta = beta; self.w = w
        (self.N, self.D) = X.shape; (self.M, _) = Y.shape
        self.max_iterations = max_iterations; self.tolerance = tolerance
        
        self.G = gaussian_kernel(self.Y, self.beta)
        self.Np = 0; self.iteration = 0

        self.diff = torch.inf; self.q = torch.inf
        if sigma2 is None: 
            self.sigma2 = compute_sigma2(self.X, self.Y, self.P)
        elif not isinstance(sigma2, torch.Tensor): 
            self.sigma2 = np2torch(sigma2)
        else: 
            self.sigma2 = sigma2
        
        self.P = torch.ones((self.M, self.N), device=self.device, requires_grad=False, dtype=torch.float64) if P is None else P
        self.Pt1 = torch.zeros((self.N, ), device=self.device, dtype=torch.float64, requires_grad=False)
        self.P1 = torch.zeros((self.M, ), device=self.device, dtype=torch.float64, requires_grad=False)
        self.PX = torch.zeros((self.M, self.D), device=self.device, dtype=torch.float64, requires_grad=False)
        self.W = torch.zeros((self.M, self.D), device=self.device, dtype=torch.float64, requires_grad=False)

        self.loss = DeformableRegistrationLoss()
        self.loss_val = None
    
    def init_state(self, iterations=3):
        ''' Used by torch solver only. Initialize P and W by running a few iterations of EM. '''
        for _ in range(iterations):
            self.expectation()
            self.maximization()

    def register(self, callback=lambda **kwargs: None):
        if self.solver == 'torch': self.init_state()

        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration, 'error': self.q, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)
        return self.TY, self.get_registration_parameters()

    def iterate(self):
        if self.solver == 'em': 
            self.expectation()
            self.maximization()
        
        else:                
            self.expectation()
            self.optimize_W()
            self.transform_point_cloud()
            # self.optimize_sigma2()
        
        self.iteration += 1

    def expectation(self):
        c = 0
        P = torch.sum((self.X[None, :, :] - self.TY[:, None, :])**2, dim=2) # (M, N)
        
        if isinstance(self.sigma2, numbers.Number):
            P = torch.exp(-P/(2*self.sigma2))
            c = (2*torch.pi*self.sigma2)**(self.D/2)*self.w/(1. - self.w)*self.M/self.N
        else:
            P = torch.exp(-P/(2*self.sigma2[:, None]))
            c = (2*torch.pi*torch.mean(self.sigma2))**(self.D/2)*self.w/(1. - self.w)*self.M/self.N

        den = torch.sum(P, axis = 0, keepdims = True) # (1, N)
        den = torch.clip(den, torch.finfo(self.X.dtype).eps, None) + c

        self.P = torch.divide(P, den)
        self.Pt1 = torch.sum(self.P, axis=0)
        self.P1 = torch.sum(self.P, axis=1)
        self.Np = torch.sum(self.P1)
        self.PX = torch.matmul(self.P, self.X)

    def maximization(self):
        self.update_transform()
        self.transform_point_cloud()
        # self.update_variance()    # FIXME: Incorporate this step after debugging update_transform()

    def update_transform(self):
        """ Calculate a new estimate of the deformable transformation.See Eq. 22 of https://arxiv.org/pdf/0905.2635.pdf."""
        
        if isinstance(self.sigma2, numbers.Number):
            A = torch.diag(self.P1) @ self.G + \
                self.alpha * self.sigma2 * torch.eye(self.M, device=self.device)
            B = self.PX - torch.diag(self.P1) @ self.Y

        else:
            dP1_inv = torch.pinverse(torch.diag(self.P1))
            A = self.G + self.alpha * (dP1_inv @ torch.diag(self.sigma2))
            B = dP1_inv @ self.PX - self.Y
        
        self.W = torch.pinverse(A) @ B
            
    def transform_point_cloud(self, Y=None):
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + G @ self.W
        else:
            self.TY = self.Y + self.G @ self.W

    def update_variance(self):
        qprev = self.sigma2

        if isinstance(self.sigma2, numbers.Number): 
            self.q = torch.inf

            xPx = torch.dot(torch.transpose(self.Pt1), torch.sum(
                torch.multiply(self.X, self.X), axis=1))
            yPy = torch.dot(torch.transpose(self.P1),  torch.sum(
                torch.multiply(self.TY, self.TY), axis=1))
            trPXY = torch.sum(torch.multiply(self.TY, self.PX))

            sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)
            if sigma2 <= 0: sigma2 = self.tolerance / 10
        
        else:   
            diff2 = torch.norm(self.TY[:,None,:] - self.X, dim=-1, p=2)**2  # (M,1,3) - (N,3) -> (M,N)
            weighted_diff2 = self.P * diff2             # (M,N)
            denom = torch.sum(self.P, axis=1)[:,None]      # (M,1)
            sigma2 = torch.sum(weighted_diff2 / denom, axis=1) / self.D

        # Here we use the difference between the current and previous
        # estimate of the variance as a proxy to test for convergence.    
        self.diff = torch.mean(torch.abs(sigma2 - qprev))
        if True or self.diff <= self.diff_bound: 
            self.sigma2 = sigma2
            self.updated_variance = True

    def get_registration_parameters(self): return self.G, self.W

    def optimize_P(self):
        ''' Solve for P using auto grad on self.loss '''
        
        self.P.requires_grad = True
        
        lr = self.optim_config['P']['lr']; epochs = self.optim_config['P']['epochs']
        optimizer = torch.optim.Adam([self.P], lr=lr)
        for epoch in (pbar:=tqdm(range(epochs), desc='Optimizing P')):
            loss = self.loss(self.X, self.Y, self.sigma2, self.P, self.G, self.W)
            pbar.set_postfix(loss=loss.item())
            optimizer.zero_grad()
            loss.backward(); optimizer.step()

        self.P.requires_grad = False
    
    def optimize_W(self):
        ''' Solve for W using auto grad on self.loss '''
        self.W.requires_grad = True

        lr = self.optim_config['W']['lr']; epochs = self.optim_config['W']['epochs']
        optimizer = torch.optim.Adam([self.W], lr=lr)
        # for epoch in (pbar:=tqdm(range(epochs), desc='Optimizing W')):
        for epoch in range(epochs):
            loss = self.loss(self.X, self.Y, self.sigma2, self.P, self.G, self.W, 
                            edges=self.mesh_generator.edges)
            optimizer.zero_grad()
            loss.backward(); optimizer.step()
            # pbar.set_postfix(loss=loss.item(), lr=lr)
        self.loss_val = loss.item()
        self.W.requires_grad = False
    
    def optimize_PW(self):
        self.P.requires_grad = True; self.W.requires_grad = True

        lr = self.optim_config['P']['lr']; epochs = self.optim_config['P']['epochs']
        optimizer = torch.optim.Adam([self.P, self.W], lr=lr)
        for epoch in (pbar:=tqdm(range(epochs), desc='Optimizing PW')):
            loss = self.loss(self.X, self.Y, self.sigma2, self.P, self.G, self.W)
            pbar.set_postfix(loss=loss.item(), lr=lr)
            optimizer.zero_grad()
            loss.backward(); optimizer.step()
        self.loss_val = loss.item()

        self.P.requires_grad = False; self.W.requires_grad = False

    
    def optimize_sigma2(self):
        ''' Solve for sigma2 using auto grad on self.loss '''
        self.sigma2.requires_grad = True

        lr = self.optim_config['sigma2']['lr']; epochs = self.optim_config['sigma2']['epochs']
        optimizer = torch.optim.Adam([self.sigma2], lr=lr)
        for epoch in (pbar:=tqdm(range(epochs), desc='Optimizing sigma2')):
            loss = self.loss(self.X, self.Y, self.sigma2, self.P, self.G, self.W)
            pbar.set_postfix(loss=loss.item())
            optimizer.zero_grad()
            loss.backward(); optimizer.step()
        self.loss_val = loss.item()

        self.sigma2.requires_grad = False