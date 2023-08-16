import torch, torch.nn as nn

class DeformableRegistrationLoss(nn.Module):
    ''' Implements loss function for deformable registration '''

    def __init__(self): 
        super(DeformableRegistrationLoss, self).__init__()
        self.losses = {}

    def forward(self, X, Y, sigma2, P, G, W, edges=None, alpha=1, beta=2):
        '''
        Inputs:
            - X: (N,D=3).       Target point cloud 
            - Y: (M,D).         Source gmm centroids
            - sigma2: (M,).     Variance of each gmm centroid
            - P: (M,N).         Soft cluster assignments
            - G: (M,M).         Y after gaussian kernel
            - W: (M,D).         Deformable transformation  \delta Y  = G @ W
            - alpha: (scalar).  Regularization strength of CPD term
            - beta: (scalar).   Regularization strength of Geodesic term
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
            
        loss = gmm_loss + alpha * cpd_loss + beta * geo_loss
        self.losses['gmm_loss'] = gmm_loss.item()
        self.losses['cpd_loss'] = cpd_loss.item()
        self.losses['geo_loss'] = geo_loss.item()
        
        return loss