import torch, numpy as np, open3d as o3d
from pytorch3d.ops import knn_points

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

class MeshGenerator:
    """ Generates mesh based on point cloud data """

    def __init__(self, points, edges=None, K=5):
        '''
        Inputs:
            - points: np.array (N,3). Point cloud data, i.e. vertices
            - edges: np.array (num_edges, 2). 
                Concretely, edges[i]=[j,k] means {points[j], points[k]} is an edge
            - K: int. Number of nearest neighbors to consider when generating mesh
        '''
        assert points.shape[-1] == 3 and len(points.shape)==2, f'Points must be of shape (N,3), got {points.shape}'
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).to(dtype=DTYPE, device=DEVICE)
        if isinstance(edges, np.ndarray): 
            edges = torch.from_numpy(edges).to(dtype=DTYPE, device=DEVICE)
        
        self.N = points.shape[0]
        self.K = K

        self.points = points; 
        self.edges = edges 

    def generate_mesh(self, method='knn'):
        """ Generates mesh based on point cloud data """
        assert method in ['knn'], f'Mesh generation method {method} not implemented'

        if method == 'knn':
            dists, indices, _ = knn_points(self.points[None,...], self.points[None,...], K=self.K+1)
            self.edges = torch.stack(
                [torch.arange(self.N, device=DEVICE)[...,None].repeat(1, self.K).flatten(),
                indices[...,1:].flatten()], dim=-1) 
            
        else: raise NotImplementedError


    def to_o3d_geometry(self):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(self.points.cpu().detach().numpy())
        line_set.lines = o3d.utility.Vector2iVector(self.edges.cpu().detach().numpy())
        return line_set