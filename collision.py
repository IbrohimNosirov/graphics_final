import taichi as ti
import taichi.math as tm
import numpy as np
vec2 = tm.vec2
vec3 = tm.vec3
vec4 = tm.vec4
vec4i = tm.ivec4
vec3i = tm.ivec3

mat3 = tm.mat3
from util import *

MAX_CONTACTS = 256
Point = ti.types.struct(v=vec2, vi=int, incI=int, ei=int, refI=int)
ContactPoints = ti.types.struct(p1=Point, p2=Point, count=int)

@ti.data_oriented
class Collision:
    def __init__(self, scene, response):
        self.scene = scene
        self.response = response
        self.init_collision()

    def init_collision(self):
        # color flag for highlighting collisions
        self.coll = ti.field(shape=(self.scene.N,), dtype=ti.u8)
        self.collPs = ti.Vector.field(2, shape=(self.scene.N * (self.scene.N + 1),), dtype=ti.f32)
        self.initCollisionPoints()

        self.num_cp = ti.field(int, shape=())
        self.num_cp[None] = 0
        

    @ti.kernel
    def initCollisionPoints(self):
        for i in range(self.scene.N * (self.scene.N + 1)):
            self.collPs[i].fill(-1)

    @ti.func
    def collide_box_halfspace(self, box, x, n):
        """
        collide box with halfspace.
        returns colliding vertex, -1 if none.
        p, q are position and orientation
        l is box dimensions
        x, n are point and outward normal defining the halfspace
        """
        sep = np.inf
        index = -1
        for i in range(self.scene.nboundary):
            s = self.scene.boundaries.p[i]
            pWorld = b2w(box.p, box.q, s*box.l/2)
            dist = (pWorld - x).dot(n) - box.rad
            if dist < sep:
                sep = dist
                index = i
        return index, sep

    @ti.func
    def find_incidentEdge(self, iv, box_inc, ref_n):
        # ref_n is in global coordinate
        # box_inc includes transformation from incident space to global space
        normal_d = np.inf
        edge = -1
        for shift in range(-1, 1):
            ie = (iv + shift) % 4
            inc_n = rot(box_inc.q, self.scene.boundaries.n[ie])
            curr_nd = tm.dot(inc_n, ref_n)
            if curr_nd < normal_d:
                edge = ie
                normal_d = curr_nd
        iv2 = (iv + 1) % 4 if edge == iv else (iv - 1) % 4
        return edge, iv2

    @ti.func
    def writeBasedOnIndex(self, out_p1: Point, out_p2: Point, p: Point, i: int):
        r_p1 = out_p1
        r_p2 = out_p2
        if i == 0:
            r_p1 = p
        elif i == 1:
            r_p2 = p
        else:
            assert (False)
        return r_p1, r_p2

    # Reference the Box2D code
    @ti.func
    def clipSegmentToRay(self, vIn: ContactPoints, normal: vec2, start_p: float, vertexI: int) -> ContactPoints:
        # Start with no output points
        count = 0

        # Calculate the distance of end points to the line
        distance0 = tm.dot(normal, vIn.p1.v) - start_p
        distance1 = tm.dot(normal, vIn.p2.v) - start_p

        out_p1 = vIn.p1
        out_p2 = vIn.p2
        # If the points are behind the plane
        if distance0 <= 0.0:
            out_p1, out_p2 = self.writeBasedOnIndex(out_p1, out_p2, vIn.p1, count)
            count += 1
        if distance1 <= 0.0:
            out_p1, out_p2 = self.writeBasedOnIndex(out_p1, out_p2, vIn.p2, count)
            count += 1

        # If the points are on different sides of the plane
        if distance0 * distance1 < 0.0:
            # Find intersection point of edge and plane
            interp = distance0 / (distance0 - distance1)
            v = vIn.p1.v + interp * (vIn.p2.v - vIn.p1.v)
            p = Point(v=v, vi=vIn.p1.vi, incI=vIn.p1.incI, ei=vIn.p1.ei, refI=vIn.p1.refI)
            if vIn.p1.vi == out_p1.vi:
                p = Point(v=v, vi=vIn.p2.vi, incI=vIn.p2.incI, ei=vIn.p2.ei, refI=vIn.p2.refI)
            out_p1, out_p2 = self.writeBasedOnIndex(out_p1, out_p2, p, count)
            count += 1
            assert (count == 2)

        return ContactPoints(p1=out_p1, p2=out_p2, count=count)

    @ti.func
    def collide_bounds(self):
        """Detect collision between box and house"""
        for i in range(self.scene.num_boxes):
            box = self.scene.boxes[i]
            # print(box)
            for j in range(self.scene.nboundary):
                boundary = self.scene.boundaries[j]
                x = boundary.p
                n = boundary.n
                iv, sep = self.collide_box_halfspace(box, x + boundary.eps * n, n)
                n_pc = 0
                is1cp = False
                is2cp = False
                r = vec2(0, 0)
                r2 = vec2(0, 0)
                x2 = vec2(0, 0)
                if sep <= 0:
                    self.coll[i] = ti.u8(255)
                    r = rot(box.q, self.scene.boundaries.p[iv] * box.l / 2)
                    cpi = ti.atomic_add(self.num_cp[None], 1)
                    self.collPs[cpi] = r + box.p
                    is1cp = True
                    n_pc += 1

                    eInc, iv2 = self.find_incidentEdge(iv, box, n)
                    x2 = b2w(box.p, box.q, self.scene.boundaries.p[iv2] * box.l / 2)

                    if tm.dot(x2 - x, n) <= boundary.eps + box.rad:
                        self.coll[i] = ti.u8(255)
                        r2 = rot(box.q, self.scene.boundaries.p[iv2] * box.l / 2)
                        cpi = ti.atomic_add(self.num_cp[None], 1)
                        self.collPs[cpi] = r2 + box.p
                        is2cp = True
                        n_pc += 1

                if is1cp:
                    self.response.addContact(x, vec2(0, 0), r, n, -1, i, tm.dot(r + box.p - x, n), 2)
                if is2cp:
                    self.response.addContact(x, vec2(0, 0), r2, n, -1, i, tm.dot(x2 - x, n), 2)

    @ti.func
    def find_max_sep(self, boxi, boxj):
        max_sep = -np.inf
        vertex_index, edge_index = -1, -1
        # check box j against edges of box i
        for ke in range(4):
            n = rot(boxi.q, self.scene.normals[ke])
            x = b2w(boxi.p, boxi.q, self.scene.normals[ke] * boxi.l / 2)
            kv, sep = self.collide_box_halfspace(boxj, x + boxi.rad * n, n)
            if sep > max_sep:
                vertex_index = kv
                edge_index = ke
                max_sep = sep

        return vertex_index, edge_index, max_sep

    @ti.func
    def collide_box_box(self, box0, box1):
        vertex_index, edge_index, max_sep = self.find_max_sep(box1, box0)
        incident_body = 0
        vertex_index1, edge_index0, max_sep1 = self.find_max_sep(box0, box1)
        if max_sep1 > max_sep:
            vertex_index = vertex_index1
            edge_index = edge_index0
            max_sep = max_sep1
            incident_body = 1
        return incident_body, vertex_index, edge_index, max_sep

    @ti.func
    def collide_all(self):
        for i in range(self.scene.num_boxes):
            for j in range(self.scene.N-1):

                if j < self.scene.num_boxes and i != j:
                    R = (self.scene.boxes[i].rad + self.scene.boxes[j].rad) * ti.sqrt(2)
                    r_sum = 0.5 * (self.scene.boxes[i].l + self.scene.boxes[j].l)
                    R = R + ti.sqrt(r_sum.x * r_sum.x + r_sum.y * r_sum.y)
                    if tm.distance(self.scene.boxes[i].p, self.scene.boxes[j].p) > R:
                        continue

                    incident_body, iv, ie, sep = self.collide_box_box(self.scene.boxes[i],
                                                                      self.scene.boxes[j])
                    if sep < 0:
                        print('Box-Box colliding')
                        self.coll[i] = ti.u8(255)
                        self.coll[j] = ti.u8(255)
                        iInc = j if incident_body else i  # index of incident body
                        iRef = i if incident_body else j  # index of reference body
                        radInc = self.scene.boxes[iInc].rad
                        radRef = self.scene.boxes[iRef].rad
                        xColl = b2w(self.scene.boxes[iInc].p,
                                    self.scene.boxes[iInc].q,
                                    self.scene.boundaries.p[iv] * self.scene.boxes[iInc].l / 2)
                        nColl = rot(self.scene.boxes[iRef].q, self.scene.boundaries.n[ie])
                        e1 = b2w(self.scene.boxes[iRef].p,
                                self.scene.boxes[iRef].q,
                                self.scene.boundaries.p[ie] * self.scene.boxes[iRef].l / 2)
                        e2 = b2w(self.scene.boxes[iRef].p, self.scene.boxes[iRef].q,
                                self.scene.boundaries.p[(ie + 1) % 4] * self.scene.boxes[iRef].l / 2)

                        eInc, iv2 = self.find_incidentEdge(iv, self.scene.boxes[iInc], nColl)
                        xColl2 = b2w(self.scene.boxes[iInc].p, self.scene.boxes[iInc].q,
                                    self.scene.boundaries.p[iv2] * self.scene.boxes[iInc].l / 2)

                        vIn = ContactPoints(p1=Point(v=xColl, vi=iv, boxID=iInc),
                                            p2=Point(v=xColl2, vi=iv2, boxID=iInc))
                        inc_tangent_12 = (e1 - e2).normalized()
                        inc_tangent_21 = -inc_tangent_12
                        offset_12 = tm.dot(e1, inc_tangent_12)
                        offset_21 = tm.dot(e2, inc_tangent_21)

                        planePoint = 0.5 * (e1 + e2)

                        clipped_ps1 = self.clipSegmentToRay(vIn, inc_tangent_12, offset_12, iv)
                        clipped_ps2 = self.clipSegmentToRay(clipped_ps1, inc_tangent_21, offset_21, iv2)
                        # assert (clipped_ps2.count == 2)

                        sep1 = (clipped_ps2.p1.v - planePoint).dot(nColl)
                        sep2 = (clipped_ps2.p2.v - planePoint).dot(nColl)
                        n_pc = 0
                        rRef1 = vec2(0, 0)
                        rInc1 = vec2(0, 0)
                        rRef2 = vec2(0, 0)
                        rInc2 = vec2(0, 0)
                        is1cp = False
                        is2cp = False
                        if sep1 <= radInc + radRef:
                            xColl1 = clipped_ps2.p1.v + 0.5 * tm.dot(planePoint - clipped_ps2.p1.v, nColl) * nColl
                            cpi = ti.atomic_add(self.num_cp[None], 1)
                            self.collPs[cpi] = xColl1
                            rInc1 = xColl1 - self.scene.boxes[iInc].p
                            rRef1 = xColl1 - self.scene.boxes[iRef].p
                            n_pc += 1
                            is1cp = True
                        if sep2 <= radInc + radRef:
                            xColl2 = clipped_ps2.p2.v + 0.5 * tm.dot(planePoint - clipped_ps2.p2.v, nColl) * nColl
                            cpi = ti.atomic_add(self.num_cp[None], 1)
                            self.collPs[cpi] = xColl2
                            rInc2 = xColl2 - self.scene.boxes[iInc].p
                            rRef2 = xColl2 - self.scene.boxes[iRef].p
                            n_pc += 1
                            is2cp = True

                        if is1cp:
                            print("box collision")
                            self.response.addContact(self.scene.boxes[iRef].p,
                                                    rRef1, rInc1, nColl, iRef, iInc, sep1, n_pc)
                        if is2cp:
                            print("box collision")
                            self.response.addContact(self.scene.boxes[iRef].p,
                                                    rRef2, rInc2, nColl, iRef, iInc, sep2, n_pc)
                            
                elif j >= self.scene.num_boxes: 
                    j_idx = j - i - 1
                    R = (self.scene.boxes[i].rad + self.scene.outer_edges[j_idx].rad) * ti.sqrt(2)
                    r_sum = 0.5 * (self.scene.boxes[i].l + self.scene.outer_edges[j_idx].l)
                    R = R + ti.sqrt(r_sum.x * r_sum.x + r_sum.y * r_sum.y)
                    if tm.distance(self.scene.boxes[i].p, self.scene.outer_edges[j_idx].p) > R:
                        continue

                    incident_body, iv, ie, sep = self.collide_box_box(self.scene.boxes[i],
                                                                    self.scene.outer_edges[j_idx])
                    if sep < 0:
                        print('colliding')
                        self.coll[i] = ti.u8(255)
                        self.coll[j] = ti.u8(255)
                        iInc = j-i if incident_body else i  # index of incident body
                        iRef = i if incident_body else j-i  # index of reference body

                        radInc = self.scene.boxes[iInc].rad
                        radRef = self.scene.boxes[iRef].rad
                        xColl = b2w(self.scene.boxes[iInc].p,
                                    self.scene.boxes[iInc].q,
                                    self.scene.boundaries.p[iv] * self.scene.boxes[iInc].l / 2)
                        nColl = rot(self.scene.boxes[iRef].q, self.scene.boundaries.n[ie])
                        e1 = b2w(self.scene.boxes[iRef].p,
                                self.scene.boxes[iRef].q,
                                self.scene.boundaries.p[ie] * self.scene.boxes[iRef].l / 2)
                        e2 = b2w(self.scene.boxes[iRef].p, self.scene.boxes[iRef].q,
                                self.scene.boundaries.p[(ie + 1) % 4] * self.scene.boxes[iRef].l / 2)

                        eInc, iv2 = self.find_incidentEdge(iv, self.scene.boxes[iInc], nColl)
                        xColl2 = b2w(self.scene.boxes[iInc].p, self.scene.boxes[iInc].q,
                                    self.scene.boundaries.p[iv2] * self.scene.boxes[iInc].l / 2)

                        vIn = ContactPoints(p1=Point(v=xColl, vi=iv, boxID=iInc),
                                            p2=Point(v=xColl2, vi=iv2, boxID=iInc))
                        inc_tangent_12 = (e1 - e2).normalized()
                        inc_tangent_21 = -inc_tangent_12
                        offset_12 = tm.dot(e1, inc_tangent_12)
                        offset_21 = tm.dot(e2, inc_tangent_21)

                        planePoint = 0.5 * (e1 + e2)

                        clipped_ps1 = self.clipSegmentToRay(vIn, inc_tangent_12, offset_12, iv)
                        clipped_ps2 = self.clipSegmentToRay(clipped_ps1, inc_tangent_21, offset_21, iv2)
                        # assert (clipped_ps2.count == 2)

                        sep1 = (clipped_ps2.p1.v - planePoint).dot(nColl)
                        sep2 = (clipped_ps2.p2.v - planePoint).dot(nColl)
                        n_pc = 0
                        rRef1 = vec2(0, 0)
                        rInc1 = vec2(0, 0)
                        rRef2 = vec2(0, 0)
                        rInc2 = vec2(0, 0)
                        is1cp = False
                        is2cp = False
                        if sep1 <= radInc + radRef:
                            xColl1 = clipped_ps2.p1.v + 0.5 * tm.dot(planePoint - clipped_ps2.p1.v, nColl) * nColl
                            cpi = ti.atomic_add(self.num_cp[None], 1)
                            self.collPs[cpi] = xColl1
                            rInc1 = xColl1 - self.scene.boxes[iInc].p
                            rRef1 = xColl1 - self.scene.boxes[iRef].p
                            n_pc += 1
                            is1cp = True
                        if sep2 <= radInc + radRef:
                            xColl2 = clipped_ps2.p2.v + 0.5 * tm.dot(planePoint - clipped_ps2.p2.v, nColl) * nColl
                            cpi = ti.atomic_add(self.num_cp[None], 1)
                            self.collPs[cpi] = xColl2
                            rInc2 = xColl2 - self.scene.boxes[iInc].p
                            rRef2 = xColl2 - self.scene.boxes[iRef].p
                            n_pc += 1
                            is2cp = True

                        if is1cp:
                            self.response.addContact(self.scene.boxes[iRef].p,
                                                    rRef1, rInc1, nColl, iRef, iInc, sep1, n_pc)
                        if is2cp:
                            self.response.addContact(self.scene.boxes[iRef].p,
                                                    rRef2, rInc2, nColl, iRef, iInc, sep2, n_pc)

    @ti.func
    def clearCollision(self):
        self.response.clearContact()
        for i in range(self.scene.N):
            self.coll[i] = ti.u8(0)

        for i in range(self.num_cp[None]):
            self.collPs[i].fill(-1)

        self.num_cp[None] = 0
