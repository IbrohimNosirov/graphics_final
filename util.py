import taichi as ti
import taichi.math as tm
import numpy as np 

# For pinning vertices / creating handle-based forces
@ti.data_oriented
class DistanceMap:
    def __init__(self, N, x):
        self.N = N
        self.x = x

        # To store mouse-vertex distance detection
        self.d = ti.field(ti.f32, N)
        self.d_temp = ti.field(ti.f32, N)
        self.di = ti.field(ti.i32, N)
        self.di_temp = ti.field(ti.i32, N)


    def get_closest_vertex(self, p: tm.vec2):
        self.fillDMap(p)
        minDN = self.N
        fromD2Dtemp = True

        while minDN > 1:
            minDN = self.DC_min(minDN, fromD2Dtemp)
            fromD2Dtemp = not fromD2Dtemp

        if fromD2Dtemp:
            min_dist = self.d[0]
            min_idx = self.di[0]
        else:
            min_dist = self.d_temp[0]
            min_idx = self.di_temp[0]

        # Random clicks that are very far away do not affect the vertices
        if min_dist > 0.2:
            min_idx = -1
        return min_idx


    # Fill a vertex-mouse distance map
    @ti.kernel
    def fillDMap(self, p: tm.vec2):
        for i in self.d:
            self.d[i] = (self.x[i] - p).dot(self.x[i] - p)
            self.di[i] = i


    # Use divide and conquer to find the minimum distance
    @ti.kernel
    def DC_min(self, arr_N: ti.i32, fromD2Dtemp: bool) -> ti.i32:
        if fromD2Dtemp:
            for i in range(0, arr_N // 2):
                self.d_temp[i] = ti.min(self.d[i * 2], self.d[i * 2 + 1])
                self.di_temp[i] = self.di[i * 2] if self.d[i * 2] < self.d[i * 2 + 1] else self.di[i * 2 + 1]

            if arr_N % 2 != 0:
                self.d_temp[arr_N // 2 - 1] = ti.min(self.d_temp[arr_N // 2 - 1], self.d[arr_N - 1])
                self.di_temp[arr_N // 2 - 1] = self.di_temp[arr_N // 2 - 1] if self.d_temp[arr_N // 2 - 1] < self.d[arr_N - 1] else self.di[
                    arr_N - 1]
        else:
            for i in range(0, arr_N // 2):
                self.d[i] = ti.min(self.d_temp[i * 2], self.d_temp[i * 2 + 1])
                self.di[i] = self.di_temp[i * 2] if self.d_temp[i * 2] < self.d_temp[i * 2 + 1] else self.di_temp[i * 2 + 1]

            if arr_N % 2 != 0:
                self.d[arr_N // 2 - 1] = ti.min(self.d[arr_N // 2 - 1], self.d_temp[arr_N - 1])
                self.di[arr_N // 2 - 1] = self.di[arr_N // 2 - 1] if self.d[arr_N // 2 - 1] < self.d_temp[arr_N - 1] else self.di_temp[arr_N - 1]

        return arr_N // 2

def get_corners(box): 
    """
    Get the four corners based off 
    """
    p = box.p
    q = box.q
    l = box.l

    ps_np = ti.Vector.field(2, dtype=ti.f32, shape=4)  # 4 vectors of 2D float32

    # the corners of the square 
    ps_np[0] = ti.Vector([p[0] - 0.5 * l[0], p[1] + 0.5 * l[1]])
    ps_np[1] = ti.Vector([p[0] - 0.5 * l[0], p[1] - 0.5 * l[1]])
    ps_np[2] = ti.Vector([p[0] + 0.5 * l[0], p[1] - 0.5 * l[1]])
    ps_np[3] = ti.Vector([p[0] + 0.5 * l[0], p[1] + 0.5 * l[1]])
    
    
    for i in range(4): 
        x = ps_np[i][0] - p[0]
        y = ps_np[i][1] - p[1]

        x_new = x * q[0] - y * q[1]
        y_new = x * q[1] + y * q[0]  # notice the + here

        ps_np[i][0] = x_new + p[0]
        ps_np[i][1] = y_new + p[1]
    

    return ps_np


vec2 = tm.vec2

@ti.func
def crossXY(u : vec2, v : vec2):
    """cross product of two xy plane vectors"""
    return u.x * v.y - u.y * v.x

@ti.func
def cross(w : float, v : vec2):
    """cross product of z-axis vector with xy-plane vector"""
    return vec2(-w*v.y, w*v.x)

@ti.func
def rot(q, X)->vec2:
    """rotate point X by rotation q"""
    return vec2(q.x * X.x - q.y * X.y, q.y * X.x + q.x * X.y)

@ti.func
def roti(q, X)->vec2:
    """rotate point X by the inverse of rotation q"""
    return vec2(q.x * X.x + q.y * X.y, -q.y * X.x + q.x * X.y)

@ti.func
def b2w(p, q, X):
    """body to world"""
    return p + rot(q, X)

@ti.func
def w2b(p, q, x):
    """world to body"""
    return roti(q, x - p)