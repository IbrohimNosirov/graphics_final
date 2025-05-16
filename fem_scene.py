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

# Scene-related Data
@ti.data_oriented
class Scene:
    def __init__(self):
        self.startScene()

    def startScene(self):
        self.init_gingerbread_box()
        self.init_boxes()

    def init_boxes(self):
        # TODO: write these as arguments
        numBoxes = 1 
        pos1 = ti.Vector([0.2,0.2])


        pos = [pos1]
        self.boxes = BoxState.field(shape=(numBoxes,))
        for i in range(numBoxes): 
            self.init_box(i, pos[i])

    
    def init_box(self, i, pos1): 
        """
        Initializes the ith box 
        """

        self.boxes[i].p = pos1
        self.boxes[i].q = ti.Vector([0, 1])
        self.boxes[i].l = ti.Vector([0.2,0.2])

       
        


    def init_box_boundaries(self):
        self.house_width = 0.9
        self.house_height = 0.6
        self.house_roof_height = 0.3
        self.house_xcenter = 0.5
        self.house_ycenter = 0.4
        
        if not hasattr(self, "boundaries"):
            self.nboundary = 5
            self.boundaries = Boundary.field(shape=(5,))
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


    @ti.kernel
    def init_boundary_indices(self):
        for i in range(5):
            self.boundary_indices[2 * i] = i
            self.boundary_indices[2 * i + 1] = (i + 1) % 5

        for i in range(4):
            self.vertex_indices[2 * i] = i
            self.vertex_indices[2 * i + 1] = (i + 1) % 4

    def init_gingerbread_box(self):
        self.init_box_boundaries()
        self.init_boundary_indices()