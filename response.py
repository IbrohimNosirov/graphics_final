import taichi as ti
import taichi.math as tm
vec2 = tm.vec2
vec3 = tm.vec3
mat3 = tm.mat3
from fem_scene import *
from util import *

MAX_CONTACTS = 256
# Contact Solver
@ti.data_oriented
class CollisionResponse:
    def __init__(self, scene, Cr, β, μ):
        self.scene = scene
        self.Cr = Cr
        self.β = β
        self.μ = μ
        self.init_contact()

# TODO: Initialize the other fields you need for implementing collision response
    def init_contact(self):
        self.num_contact = ti.field(shape=(), dtype=ti.i32)
        self.p1 = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTACTS,))
        self.i1 = ti.field(dtype=ti.i32, shape=(MAX_CONTACTS,))
        self.i2 = ti.field(dtype=ti.i32, shape=(MAX_CONTACTS,))
        self.n1 = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTACTS,))
        self.r1 = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTACTS,))
        self.r2 = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTACTS,))
        self.sep =ti.field(dtype=ti.f32, shape=(MAX_CONTACTS,))
        # PGS state
        self.omega_n = ti.field(dtype=ti.f32, shape=(MAX_CONTACTS,))
        self.impulse_change = ti.field(dtype=ti.f32, shape=())
        self.dv = ti.Vector.field(2, dtype=ti.f32, shape=(self.scene.N,))
        self.dw = ti.field(dtype=ti.f32, shape=(self.scene.N,))

        self.num_contact[None] = 0
        for i in range(self.scene.N):
            self.dv[i] = vec2(0.0, 0.0)
            self.dw[i] = 0.0
        for c_idx in range(MAX_CONTACTS):
             self.omega_n[c_idx] = 0.0

    @ti.func
    def pgs_iteration(self):
        self.impulse_change[None] = 0.0
        for c_idx in range(self.num_contact[None]):
            iA, iB = self.i1[c_idx], self.i2[c_idx]
            rA, rB = self.r1[c_idx], self.r2[c_idx]
            n, sep = self.n1[c_idx], self.sep[c_idx]
            old_omega = self.omega_n[c_idx]

            # A (Reference)
            vA, wA, invMA, invIA = vec2(0.0, 0.0), 0.0, 0.0, 0.0
            if iA >= 0: # if not boundary
                boxA = self.scene.boxes[iA]
                vA = boxA.v + self.dv[iA]
                wA = boxA.ω + self.dw[iA]
                invMA = 1.0 / boxA.m if boxA.m > 1e-12 else 0.0
                invIA = 1.0 / boxA.I if boxA.I > 1e-12 else 0.0

            # B (Incident)
            vB, wB, invMB, invIB = vec2(0.0, 0.0), 0.0, 0.0, 0.0
            if iB >= 0:
                boxB = self.scene.boxes[iB]
                vB = boxB.v + self.dv[iB]
                wB = boxB.ω + self.dw[iB]
                invMB = 1.0 / boxB.m if boxB.m > 1e-12 else 0.0
                invIB = 1.0 / boxB.I if boxB.I > 1e-12 else 0.0

            # B rel A is B-A = (vB + cross(wB, rB)) - (vA + cross(wA, rA))
            relV = (vB + cross(wB, rB)) - (vA + cross(wA, rA))
            v_n = relV.dot(n)

            # K = J M^-1 J^T = sum(m_inv) + sum((r x n)^2 * I_inv)
            termA_rot = (crossXY(rA, n)**2) * invIA if iA >= 0 else 0.0
            termB_rot = (crossXY(rB, n)**2) * invIB if iB >= 0 else 0.0
            m_eff = 1 / (invMA + invMB + termA_rot + termB_rot)

            # rel normal velocity after impulse
            vA_initial, wA_initial = vec2(0.0, 0.0), 0.0
            if iA >= 0: 
                vA_initial = self.scene.boxes[iA].v
                wA_initial = self.scene.boxes[iA].ω
            vB_initial, wB_initial = vec2(0.0, 0.0), 0.0
            if iB >= 0: 
                vB_initial = self.scene.boxes[iB].v
                wB_initial = self.scene.boxes[iB].ω
            relV_initial = (vB_initial + cross(wB_initial, rB)) - (vA_initial + cross(wA_initial, rA))
            v_n_initial = relV_initial.dot(n)

            # max(pentration correction, restitution)
            v_target = tm.max(self.β * ti.max(-sep, 0.0), -self.Cr * v_n_initial)
            v_target = tm.max(v_target, 0.0)

            # omega increment
            delta_omega = (v_target - v_n) * m_eff
            new_omega = ti.max(old_omega + delta_omega, 0.0)
            # fix delta_omega from clamping
            delta_omega = new_omega - old_omega
            self.omega_n[c_idx] = new_omega

            if abs(delta_omega) > 1e-12:
                impulse_vec = delta_omega * n
                if iA >= 0:
                    self.dv[iA] -= impulse_vec * invMA
                    self.dw[iA] -= invIA * crossXY(rA, impulse_vec)
                if iB >= 0:
                    self.dv[iB] += impulse_vec * invMB
                    self.dw[iB] += invIB * crossXY(rB, impulse_vec)

                self.impulse_change[None] += abs(delta_omega)

    @ti.func
    def PGS(self):
        i = 0
        while self.impulse_change[None] < 1e-6 and i < 100:
            self.pgs_iteration()
            i += 1

    @ti.func
    def apply_impulses(self):
        for i in range(self.scene.N):
            if self.scene.boxes[i].m > 1e-12:
                 self.scene.boxes[i].v += self.dv[i]
            if self.scene.boxes[i].I > 1e-12:
                 self.scene.boxes[i].ω += self.dw[i]

    @ti.func
    def addContact(self, p1: vec2, r1: vec2, r2: vec2, n1: vec2, i1: int, i2: int, sep:float, nc: int):
        """
        This function is being triggered after the
        :param p1: vec2, the mass center of the reference rigid body
        :param r1: vec2, the displacement from the reference rigid body mass center to the contact point
        :param r2: vec2, the displacement from the incident rigid body mass center to the contact point
        :param n1: vec2, the normal of the reference edge
        :param i1: int, the index of the reference rigid body. You may find info related to this rigid body in self.state.boxes[i1]
        :param i2: int, the index of the incident rigid body. You may find info related to this rigid body in self.state.boxes[i2]
        :param sep: float, the maximum seperation distance between two boxes
        :param nc: int, number of contact points in between body i1 and i2.
        :return: void
        """
        # Note that if i1 < 0, the reference rigid body could be a rigid line boundary. Then, p1 would be a point on the
        # line boundary, r1 would be vec2(0, 0) and n1 would be the normal of the line boundary
        if self.num_contact[None] < MAX_CONTACTS:
            c_id = self.num_contact[None]
            self.p1[c_id] = p1
            self.i1[c_id] = i1
            self.i2[c_id] = i2
            self.n1[c_id] = n1
            self.r1[c_id] = r1
            self.r2[c_id] = r2
            self.sep[c_id] = sep
            self.num_contact[None] += 1
            print("number of contacts ", self.num_contact[None])

    @ti.func
    def clearContact(self):
        self.num_contact[None] = 0
        for i in range(self.scene.N):
            self.dv[i] = vec2(0.0, 0.0)
            self.dw[i] = 0.0
        for c_idx in range(MAX_CONTACTS):
             self.omega_n[c_idx] = 0.0