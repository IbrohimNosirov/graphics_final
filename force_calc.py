import taichi as ti
import taichi.math as tm

@ti.func
def compute_D(x0, x1, x2):
    """
    Compute the edge matrix D from the triangle vertex positions.
    """
    D = ti.Matrix.zero(dt=ti.f32, n=2, m=2)
    D[:, 0] = x1 - x0
    D[:, 1] = x2 - x0
    return D

@ti.func
def compute_F(D, D0_inv):
    """
    Compute the deformation gradient F from the current edge matrix D
    and the rest-configuration matrix D0.
    """
    return D @ D0_inv

@ti.func
def compute_P_NeoHookean(F, Lambda, Mu):
    """
    Compute the first Piola-Kirchhoff stress for the Neo-Hookean model.
    """
    inv_F_T = tm.inverse(F).transpose()
    P = Mu * (F - inv_F_T) + Lambda * tm.log(tm.determinant(F)) * inv_F_T
    return P

@ti.func
def compute_P_StVK(F, Lambda, Mu):
    """
    Compute the first Piola-Kirchhoff stress for the St. Venant-Kirchhoff model.
    """
    E = 0.5 * (F.transpose() @ F - tm.eye(2))
    P = F @ (Lambda * E.trace() * tm.eye(2) + 2 * Mu * E)
    return P

@ti.func
def polar_decompose(F):
    """
    Perform polar decomposition of F.
    Returns the rotation R and the symmetric component S.
    """
    U, Sigma, V = ti.svd(F)
    det_U = tm.determinant(U)
    det_V = tm.determinant(V)
    L = tm.eye(2)
    L[1, 1] = det_U * det_V
    # Using ti.select for conditional operations
    U = ti.select((det_U < 0) and (det_V > 0), U @ L, U)
    V = ti.select((det_U > 0) and (det_V < 0), V @ L, V)

    Sigma = Sigma @ L
    R = U @ V.transpose()
    S = V @ Sigma @ V.transpose()
    return R, S

@ti.func
def compute_P_Corotated(F, Lambda, Mu):
    """
    Compute the first Piola-Kirchhoff stress for the corotational linear model.
    """
    R, S = polar_decompose(F)
    eps = S - tm.eye(2)
    P = R @ (2 * Mu * eps + Lambda*eps.trace() * tm.eye(2)) 
    return P

@ti.func
def compute_H(D0_inv, P, A):
    """
    Compute the force matrix H given the rest configuration D0 and stress P.
    Returns a list [f0, f1, f2] where f0, f1, f2 are the forces on each triangle vertex.
    """
    # Compute the (signed) area of the triangle in its rest configuration.
    H = -A * P @ D0_inv.transpose()
    # The forces on the triangle vertices:
    f1 = H[:, 0]
    f2 = H[:, 1]
    f0 = -(f1 + f2)
    return [f0, f1, f2]

@ti.func
def update_forces(force, vertices, forces_tri):
    """
    Distribute triangle forces to the global force field.
    'vertices' is a 3-element vector of vertex indices,
    and 'forces_tri' is a list [f0, f1, f2] of computed forces.
    """
    force[vertices[0]] += forces_tri[0]
    force[vertices[1]] += forces_tri[1]
    force[vertices[2]] += forces_tri[2]