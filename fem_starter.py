import taichi as ti
import taichi.math as tm
import numpy as np
from util import *
from pywavefront import Wavefront
from fem_scene import Scene

# Import force calculation functions from the force_calc module
from force_calc import (
    compute_D, compute_F, compute_P_NeoHookean, compute_H, update_forces)

# ti.init(arch=ti.cpu, debug=True)

ti.init(arch=ti.vulkan)

## physical quantities
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
YoungsModulus[None] = 1e3
PoissonsRatio[None] = 0
gravity = ti.Vector([0, -9.8])
kspring = 100

##############################################################
# Lame parameters
Lambda = ti.field(ti.f32, ())
Lambda[None] = YoungsModulus[None]*PoissonsRatio[None] / ((1+PoissonsRatio[None])*(1-PoissonsRatio[None]))
Mu = ti.field(ti.f32, ())
Mu[None] = YoungsModulus[None]/(2*(1 + PoissonsRatio[None]))
##############################################################

## Load geometry
obj = Wavefront('models/woody-halfres.obj', collect_faces=True)
va = np.array(obj.vertices, dtype=np.float32)[:,:2] * 0.8
x_avg = 0.5*(np.amin(va[:,0])+np.amax(va[:,0]))
y_avg = 0.5*(np.amin(va[:,1])+np.amax(va[:,1]))
va += np.array([0.5-x_avg, 0.5-y_avg])
mesh = obj.mesh_list[0]
faces = np.array(mesh.faces, dtype=np.int32)
mesh_triangles = ti.field(int, shape=np.prod(faces.shape))
mesh_triangles.from_numpy(faces.ravel())

# Number of triangles and vertices
N_triangles = faces.shape[0]
triangles = ti.Vector.field(3, ti.i32, N_triangles)
for i in range(N_triangles):
    triangles[i] = ti.Vector(faces[i])

edges_set = set()
tmp_set = set()
for i in range(N_triangles):
    element1 = (faces[i][0],faces[i][1]) if faces[i][0] < faces[i][1] else (faces[i][1], faces[i][0])
    element2 = (faces[i][1],faces[i][2]) if faces[i][1] < faces[i][2] else (faces[i][2], faces[i][1])
    element3 = (faces[i][2],faces[i][0]) if faces[i][2] < faces[i][0] else (faces[i][0], faces[i][2])
    for element in [element1, element2, element3]:
        if element in edges_set:
            tmp_set.add(element)
        else:
            edges_set.add(element)

outer_edges_set = edges_set - tmp_set

N_edges = len(edges_set)
np_edges = np.array([list(e) for e in edges_set])
edges = ti.Vector.field(2, shape=N_edges, dtype=int)
edges.from_numpy(np_edges)

N_outer_edges = len(outer_edges_set)
np_outer_edges = np.array([list(e) for e in outer_edges_set])
outer_edges = ti.Vector.field(2, shape=N_outer_edges, dtype=int)
outer_edges.from_numpy(np_outer_edges)


# Create house scene for collision
scene = Scene(outer_edges, va)

#############################################################
# Simulation parameters
h = 16.7e-3
substepping = 100
dh = h/substepping
N = va.shape[0]
m = 1/N*25 

# Simulation fields
x = ti.Vector.field(2, shape=N, dtype=ti.f32)
x.from_numpy(va)
v = ti.Vector.field(2, ti.f32, N)
force = ti.Vector.field(2, ti.f32, N)
force_idx = ti.field(ti.i32, ())
spring_force = ti.Vector.field(2, ti.f32, 1)
force_start_pos = np.array([0,0])
force_end_pos = np.array([0,0])
draw_force = False

# Define D0_inv per triangle
D0_inv = ti.Matrix.field(2, 2, shape=N_triangles, dtype=ti.f32)
Area = ti.field(ti.f32, shape=N_triangles)

@ti.func
def get_vertices(i):
    vertices = triangles[i]
    return x[vertices[0]], x[vertices[1]], x[vertices[2]]

# Initialize rest configuration for each triangle
@ti.kernel
def init_D0_inv():
    for i in range(N_triangles):
        a, b, c = get_vertices(i)
        D0 = compute_D(a, b, c)
        Area[i] = 0.5 * tm.determinant(D0)
        D0_inv[i] = tm.inverse(D0)


@ti.func 
def update_box_position():
    for i in range(scene.boxes.shape[0]): 
        scene.boxes[i].p += dh * scene.boxes[i].v
        # scene.boxes[i].q += dh * scene.boxes[i].ω
        # scene.boxes[i].q = tm.length()


# Timestep kernel using NeoHookean model only
@ti.kernel
def timestep():
    # Clear forces over triangles (force is per vertex)
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])
    
    # Compute internal forces per triangle using NeoHookean model
    for i in range(N_triangles):
        verts = triangles[i]
        # Get current vertex positions
        a = x[verts[0]]
        b = x[verts[1]]
        c = x[verts[2]]
        # Compute the current deformation edge matrix
        D = compute_D(a, b, c)
        # Compute the deformation gradient
        F_val = compute_F(D, D0_inv[i])
        # Use NeoHookean model only
        P = compute_P_NeoHookean(F_val, Lambda[None], Mu[None])
        # Compute force contribution
        H_val = compute_H(D0_inv[i], P, Area[i])
        # Distribute force to the triangle vertices
        update_forces(force, verts, H_val)
    
    # Update velocities (Simplectic Euler update)
    for i in range(N):
        v[i] += dh*force[i]/m
        
    # Apply spring force if dragging
    cur_spring_len = tm.sqrt(tm.dot(spring_force[0], spring_force[0]))
    if force_idx[None] > 0:
        v[force_idx[None]] += dh*kspring*spring_force[0]/(m)*cur_spring_len
    
    # Handle collision with house boundaries
    for i in range(N):
        for j in range(scene.boundaries.p.shape[0]):
            b = scene.boundaries.p[j]
            eps = scene.boundaries.eps[j]
            n = scene.boundaries.n[j]
            
            next_pos = x[i] + dh * v[i]
            dist = (next_pos - b).dot(n)
            
            if dist < eps:
                v_n = v[i].dot(n) * n
                v_t = v[i] - v_n       
                if v_n.dot(n) < 0:     
                    v[i] = v_t
    
    update_box_position()

    # Update positions
    for i in range(N):
        x[i] += dh * v[i]

##############################################################

@ti.kernel
def reset_user_drag():
    force_idx[None] = -1
    spring_force[0] = ti.Vector([0,0])

def reset_state():
    x.from_numpy(va)
    for i in range(N):
        v[i] = ti.Vector([0,0])
    reset_user_drag()

# Initialize
init_D0_inv()
dm = DistanceMap(N, x)
window = ti.ui.Window("NeoHookean FEM with House Collision", (600, 600))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))

# For drawing the line indicating force direction/magnitude
draw_force_vertices = ti.Vector.field(2, shape=2, dtype=ti.f32)
draw_force_indices = ti.Vector.field(2, shape=1, dtype=int)
draw_force_indices[0] = ti.Vector([0,1])

while window.running:
    if window.is_pressed(ti.ui.LMB):
        if not draw_force:
            draw_force = True
            force_idx[None] = dm.get_closest_vertex(ti.Vector(window.get_cursor_pos()))
            if force_idx[None] > 0:
                force_start_pos = np.array([x[force_idx[None]][0], x[force_idx[None]][1]])
                force_end_pos = force_start_pos
        else:
            if force_idx[None] > 0:
                force_start_pos = np.array([x[force_idx[None]][0], x[force_idx[None]][1]])
                force_end_pos = np.array(window.get_cursor_pos())
                spring_force[0] = ti.Vector(force_end_pos-force_start_pos)
                draw_force_vertices.from_numpy(
                        np.stack((force_start_pos,force_end_pos))
                        .astype(np.float32))
                canvas.lines(vertices=draw_force_vertices,
                        indices=draw_force_indices,
                        color=(1,0,0),width=0.002)
        
    for e in window.get_events(ti.ui.RELEASE):
        if e.key == ti.ui.LMB:
            reset_user_drag()
            draw_force = False

    for e in window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'r' or e.key == 'R':
            reset_state()

    # Run simulation steps
    for i in range(substepping):
        timestep()

    # Draw wireframe of mesh
    canvas.lines(vertices=x, indices=edges, width=0.002, color=(0,0,0))

    # Draw the gingerbread house
    canvas.lines(scene.boundaries.p, width=0.01, indices=scene.boundary_indices, color=(0.4, 0.2, 0.0))

    # Draw the boxes 
    for i in range(scene.boxes.shape[0]):
        vertices = get_corners(scene.boxes[i])
        canvas.lines(vertices, width=0.01, indices=scene.vertex_indices, color=(0.4, 0.2, 0.0))

    # GUI text
    gui = window.get_gui()
    with gui.sub_window("Controls", 0.02, 0.02, 0.4, 0.15):
        gui.text('NeoHookean model with house collision')
        gui.text('Drag to move the deformable object')
        gui.text('Press \'r\' to reset to initial state')

    window.show()