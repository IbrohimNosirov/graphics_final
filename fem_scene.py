import taichi as ti
import taichi.math as tm

import numpy as np

vec2 = tm.vec2
vec3 = tm.vec3
vec3i = ti.types.vector(3, int)

@ti.dataclass
class Boundary:
    p: vec2
    n: vec2
    v: vec2
    eps:float

# also add a BoxState class
@ti.dataclass
class BoxState: 
    p : vec2  # position (center of mass)
    q : vec2  # orientation (cosine/sine pair)
    v : vec2  # linear velocity
    ω : float # angular velocity
    l : vec2  # dimensions of box
    m : float # mass
    I : float # moment of inertia
    rad: float # collision detection radius
    n: vec2 # normal vector 
    eps: float # distance to the boundary
    is_mesh : bool 

@ti.dataclass
class Particle:
    p: vec2
    v: vec2
    r: float
    eps: float
    m: float

# Scene-related Data
@ti.data_oriented
class Scene:
    def __init__(self, outer_edges, edge_vertices):
        # data structures
        self.N_outer_edges = outer_edges.shape[0]
        self.num_boxes = 1
        self.N = self.N_outer_edges + self.num_boxes
        self.outer_edges = BoxState.field(shape=(self.N_outer_edges,))
        self.boxes = BoxState.field(shape=(self.num_boxes,))
        self.nboundary = 5 # a house has 5 edges.
        self.boundaries = Boundary.field(shape=(self.nboundary,))
        self.normals = ti.Vector.field(2, shape=(4,), dtype=ti.f32)
        self.normals.from_numpy(np.array([[0,-1], [1,0], [0,1], [-1,0]], dtype=np.float32))
        self.num_particles = 1 
        self.circles = Particle.field(shape=(self.num_particles,))


        # initialize
        # self.init_outer_edges(outer_edges, edge_vertices)
        self.init_gingerbread_box()
        self.init_boxes()
        self.init_particles()

    def init_particles(self): 
        """
        Initializes the particles in the scene. 
        """
        for i in range(self.num_particles):
            p = ti.Vector([0.2, 0.2])
            self.circles[i].p = p
            self.circles[i].r = 0.05
            self.circles[i].v = ti.Vector([0.0, 0.0])
        self.init_particles_indices()

    def init_particles_indices(self): 
        """
        Initializes the particles indices. 
        """
        self.particle_indices = ti.field(shape=(50 * self.num_particles,), dtype=ti.i32)
        for i in range(self.num_particles):
            for j in range(50):
                self.particle_indices[50 * i + j] = j


    def init_outer_edges(self, outer_edges, edge_vertices):
        for i in range(self.N_outer_edges):
            vertex_pair = outer_edges[i]
            v1 = edge_vertices[vertex_pair[0]]
            v2 = edge_vertices[vertex_pair[1]]
            eps = 1.0e-4
            self.outer_edges[i].p = (v1 + v2)/2
            self.outer_edges[i].q = v1 - v2
            self.outer_edges[i].l = ti.Vector([np.linalg.norm(v1 - v2), eps])
            self.outer_edges[i].m = 0.1
            self.outer_edges[i].I = (1/12) * self.outer_edges[i].m\
                                    * self.outer_edges[i].l.dot(self.outer_edges[i].l)
            self.outer_edges[i].rad = eps
            self.outer_edges[i].is_mesh = True

    def init_gingerbread_box(self):
        self.init_box_boundaries()
        self.init_boundary_indices()

    def init_boxes(self):
        # TODO: write these as arguments
        pos1 = ti.Vector([0.2,0.2])
        pos2 = ti.Vector([0.45,0.2])

        pos = [pos1, pos2]
        for i in range(self.num_boxes): 
            self.init_box(i, pos[i])
    
    def init_box(self, i, pos1): 
        """
        Initializes the ith box 
        """
        theta = 0
        self.boxes[i].p = pos1
        self.boxes[i].q = ti.Vector([tm.cos(theta), tm.sin(theta)])
        self.boxes[i].l = ti.Vector([0.1,0.1])
        self.boxes[i].v = ti.Vector([0.0, 0.1])
        self.boxes[i].ω = 3
        self.boxes[i].is_mesh = False 





    def init_box_boundaries(self):
        self.house_width = 0.7
        self.house_height = 0.6
        self.house_roof_height = 0.3
        self.house_xcenter = 0.5
        self.house_ycenter = 0.4
        
        ps_np = np.array([  [self.house_xcenter - 0.5 * self.house_width, self.house_ycenter + 0.5 * self.house_height],
                            [self.house_xcenter - 0.5 * self.house_width, self.house_ycenter - 0.5 * self.house_height],
                            [self.house_xcenter + 0.5 * self.house_width, self.house_ycenter - 0.5 * self.house_height],
                            [self.house_xcenter + 0.5 * self.house_width, self.house_ycenter + 0.5 * self.house_height],
                            [self.house_xcenter, self.house_ycenter + 0.5 * self.house_height + self.house_roof_height]],
                            dtype=np.float32)
        rooftop_right = (ps_np[4] - ps_np[3]) / np.linalg.norm(ps_np[4] - ps_np[3])
        rooftop_left = (ps_np[0] - ps_np[4]) / np.linalg.norm(ps_np[0] - ps_np[4])

        self.boundaries.p.from_numpy(ps_np)
        self.boundaries.n.from_numpy(np.array([[1, 0],
                                                [0, 1],
                                                [-1, 0],
                                                [-rooftop_right[1], rooftop_right[0]], 
                                                [-rooftop_left[1], rooftop_left[0]]], dtype=np.float32))
        self.boundaries.eps.from_numpy(np.ones(5,  dtype=np.float32) * 1e-2)
        self.boundary_indices = ti.field(shape=(10,), dtype=ti.i32)
        self.vertex_indices = ti.field(shape=(8,), dtype=ti.i32)
        self.boundaries.v.from_numpy(np.zeros((5,2), dtype=np.float32))

    @ti.kernel
    def init_boundary_indices(self):
        for i in range(5):
            self.boundary_indices[2 * i] = i
            self.boundary_indices[2 * i + 1] = (i + 1) % 5

        for i in range(4):
            self.vertex_indices[2 * i] = i
            self.vertex_indices[2 * i + 1] = (i + 1) % 4
