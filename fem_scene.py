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

# Scene-related Data
@ti.data_oriented
class Scene:
    def __init__(self):
        self.startScene()

    def startScene(self):
        self.init_gingerbread_box()

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

    @ti.kernel
    def init_boundary_indices(self):
        for i in range(5):
            self.boundary_indices[2 * i] = i
            self.boundary_indices[2 * i + 1] = (i + 1) % 5

    def init_gingerbread_box(self):
        self.init_box_boundaries()
        self.init_boundary_indices()