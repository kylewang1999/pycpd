import torch, numpy as np, open3d as o3d
from pytorch3d.ops import knn_points

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

class MeshGenerator:
    """ Generates mesh based on point cloud data """

    def __init__(self, points, K=5, method='knn'):
        '''
        Inputs:
            - points: np.array (N,3). Point cloud data, i.e. vertices
            - edges: np.array (E, 2). 
                Concretely, edges[i]=[j,k] means {points[j], points[k]} is an edge
            - K: int. Number of nearest neighbors to consider when generating mesh
        Note: There are 3 ways to represent as mesh in MeshGenerator
            1. self.edges (E,2) (when mesh is not strictly triangular). Stores a list of edges, where each edge is represented as two vertices [j,k]
            2. self.faces (F,3) (when mesh is triangular). Stores a list of faces, where each face is represented as [j,k,l]
            3. self.mesh. Open3d TriangleMesh object. 
        '''
        assert points.shape[-1] == 3 and len(points.shape)==2, f'Points must be of shape (N,3), got {points.shape}'
        assert method in ['knn', 'poisson', 'alpha_shape', 'ball_pivoting'], f'Mesh generation method {method} not implemented'

        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).to(dtype=DTYPE, device=DEVICE)
   
        self.method = method
        self.N = points.shape[0]
        self.K = K

        self.points = points; 
        self.edges, self.edge_lens, self.faces, self.mesh = None, None, None, None
        self.generate_mesh()

    def generate_mesh(self):
        """ Generates mesh based on point cloud data """
        
        if self.method == 'knn':
            dists, indices, _ = knn_points(self.points[None,...], self.points[None,...], K=self.K+1)
            edges = torch.stack(
                [torch.arange(self.N, device=DEVICE)[...,None].repeat(1, self.K).flatten(),
                indices[...,1:].flatten()], dim=-1) 
            self.edges = edges
            self.edge_lens = torch.norm(
                self.points[edges[...,0]] - self.points[edges[...,1]], 
                dim=-1, p=2)
            
        elif self.method in ['poisson', 'alpha_shape', 'ball_pivoting']:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(self.points.cpu().detach().numpy())
            pcd_o3d.estimate_normals()

            if self.method == 'poisson':
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_o3d, depth=12)
            elif self.method == 'alpha_shape':
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_o3d, alpha=1e-2)
            elif self.method == 'ball_pivoting':
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_o3d, 
                    radii=o3d.utility.DoubleVector([5e-4, 1e-3, 1e-2, 2e-2, 8e-2, 1e-1]))
            mesh.compute_vertex_normals()
            
            self.mesh = mesh
        
        else: raise NotImplementedError


    def to_o3d_geometry(self):
        ''' Convert self.edges or self.faces or self.mesh to o3d.geometry for visualization
        Outputs:
            - Open3d geometry object that represent the mesh
        '''
        if self.edges is not None:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(self.points.cpu().detach().numpy())
            line_set.lines = o3d.utility.Vector2iVector(self.edges.cpu().detach().numpy())
            return line_set
        
        elif self.mesh is not None:
            assert isinstance(self.mesh, o3d.geometry.TriangleMesh), f'Mesh must be of type o3d.geometry.TriangleMesh, got {type(self.mesh)}'
            verts = o3d.utility.Vector3dVector(self.mesh.vertices)
            tris = o3d.utility.Vector3iVector(self.mesh.triangles)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = verts
            mesh.triangles = tris
            return self.mesh

        else: return None