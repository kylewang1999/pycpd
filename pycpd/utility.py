import numpy as np, torch


def is_positive_semi_definite(R):
    if not isinstance(R, (np.ndarray, np.generic)):
        raise ValueError('Encountered an error while checking if the matrix is positive semi definite. \
            Expected a numpy array, instead got : {}'.format(R))
    return np.all(np.linalg.eigvals(R) > 0)

def gaussian_kernel(X, beta, Y=None, use_torch=False):
    if Y is None:
        Y = X
    diff = (X[:, None, :] - Y[None, :,  :]) ** 2
    diff = diff.sum(dim=2) if use_torch else np.sum(diff, 2)
    return torch.exp(-diff / (2 * beta**2)) if use_torch else np.exp(-diff / (2 * beta**2))

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
