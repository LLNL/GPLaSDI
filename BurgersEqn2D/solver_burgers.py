import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from time import time



def solver(ic, amp, width, Re, nx, ny, nt, dt, nxy, dx, dy, maxitr, tol):

    '''

    Solves 2D Burgers equation for generating the data
    from https://github.com/LLNL/gLaSDI and https://github.com/LLNL/LaSDI

    '''

    if ic == 1:  # sine
        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1
    elif ic == 2:  # Gaussian
        xmin = -3
        xmax = 3
        ymin = -3
        ymax = 3
        x0 = 0  # Gaussian center
        y0 = 0  # Gaussian center
    else:
        print('wrong values for IC!')
    I = sp.eye(nxy, format='csr')

    # full indicies, free indicies, fixed indicies
    [xv, yv] = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny), indexing='xy')
    x = xv.flatten()
    y = yv.flatten()

    multi_index_i, multi_index_j = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    full_multi_index = (multi_index_j.flatten(), multi_index_i.flatten())
    free_multi_index = (multi_index_j[1:-1, 1:-1].flatten(), multi_index_i[1:-1, 1:-1].flatten())
    x0_multi_index = (multi_index_j[1:-1, 0].flatten(), multi_index_i[1:-1, 0].flatten())
    x1_multi_index = (multi_index_j[1:-1, -1].flatten(), multi_index_i[1:-1, -1].flatten())
    y0_multi_index = (multi_index_j[0, 1:-1].flatten(), multi_index_i[0, 1:-1].flatten())
    y1_multi_index = (multi_index_j[-1, 1:-1].flatten(), multi_index_i[-1, 1:-1].flatten())

    dims = (ny, nx)
    full_raveled_indicies = np.ravel_multi_index(full_multi_index, dims)
    free_raveled_indicies = np.ravel_multi_index(free_multi_index, dims)
    x0_raveled_indicies = np.ravel_multi_index(x0_multi_index, dims)
    x1_raveled_indicies = np.ravel_multi_index(x1_multi_index, dims)
    x01_raveled_indicies = np.concatenate((x0_raveled_indicies, x1_raveled_indicies))
    y0_raveled_indicies = np.ravel_multi_index(y0_multi_index, dims)
    y1_raveled_indicies = np.ravel_multi_index(y1_multi_index, dims)
    y01_raveled_indicies = np.concatenate((y0_raveled_indicies, y1_raveled_indicies))
    fixed_raveled_indicies = np.setdiff1d(full_raveled_indicies, free_raveled_indicies)

    # boundary one-hot vector
    x0_one_hot = np.eye(nx - 2)[0]
    y0_one_hot = np.eye(ny - 2)[0]
    x1_one_hot = np.eye(nx - 2)[-1]
    y1_one_hot = np.eye(ny - 2)[-1]

    # inner grid
    inner_multi_index_i, inner_multi_index_j = np.meshgrid(np.arange(nx - 2), np.arange(ny - 2), indexing='xy')
    inner_x_multi_index = (np.concatenate((inner_multi_index_j[:, 0].flatten(), inner_multi_index_j[:, -1].flatten())),
                           np.concatenate((inner_multi_index_i[:, 0].flatten(), inner_multi_index_i[:, -1].flatten())))
    inner_y_multi_index = (np.concatenate((inner_multi_index_j[0, :].flatten(), inner_multi_index_j[-1, :].flatten())),
                           np.concatenate((inner_multi_index_i[0, :].flatten(), inner_multi_index_i[-1, :].flatten())))

    inner_dims = (ny - 2, nx - 2)
    inner_x_raveled_indicies = np.ravel_multi_index(inner_x_multi_index, inner_dims)
    inner_y_raveled_indicies = np.ravel_multi_index(inner_y_multi_index, inner_dims)

    # first order derivative
    # central
    Mcb = sp.diags([np.zeros(nx - 2), -np.ones(nx - 2), np.ones(nx - 2)], [0, -1, 1], (nx - 2, nx - 2))
    Mc = sp.kron(sp.eye(ny - 2), Mcb, format="csr")

    Ib = sp.eye(nx - 2)
    Nc = sp.kron(sp.diags([np.zeros(ny - 2), -np.ones(ny - 2), np.ones(ny - 2)], [0, -1, 1], (ny - 2, ny - 2)), Ib,
                 format="csr")

    # forward
    Mfb = sp.diags([-np.ones(nx - 2), np.ones(nx - 2)], [0, 1], (nx - 2, nx - 2))
    Mf = sp.kron(sp.eye(ny - 2), Mfb, format="csr")

    Ib = sp.eye(nx - 2)
    Nf = sp.kron(sp.diags([-np.ones(ny - 2), np.ones(ny - 2)], [0, 1], (ny - 2, ny - 2)), Ib, format="csr")

    # backward
    Mbb = sp.diags([np.ones(nx - 2), -np.ones(nx - 2)], [0, -1], (nx - 2, nx - 2))
    Mb = sp.kron(sp.eye(ny - 2), Mbb, format="csr")

    Ib = sp.eye(nx - 2)
    Nb = sp.kron(sp.diags([np.ones(ny - 2), -np.ones(ny - 2)], [0, -1], (ny - 2, ny - 2)), Ib, format="csr")

    # laplacian operator
    Dxb = sp.diags([-2 * np.ones(nx - 2), np.ones(nx - 2), np.ones(nx - 2)], [0, -1, 1], (nx - 2, nx - 2))
    Dx = sp.kron(sp.eye(ny - 2), Dxb, format="csr")

    Ib = sp.eye(nx - 2)
    Dy = sp.kron(sp.diags([-2 * np.ones(ny - 2), np.ones(ny - 2), np.ones(ny - 2)], [0, -1, 1], (ny - 2, ny - 2)), Ib,
                 format="csr")

    timer = []
    timer.append(time())

    # compute u_full and v_full
    if ic == 1:  # IC: sine
        zv = amp * np.sin(2 * np.pi * xv) * np.sin(2 * np.pi * yv)
        zv[np.nonzero(xv > 0.5)] = 0.0
        zv[np.nonzero(yv > 0.5)] = 0.0
    elif ic == 2:  # IC: Gaussian
        zv = amp * np.exp(-((xv - x0) ** 2 + (yv - y0) ** 2) / width)
        z = zv.flatten()
    u0 = z.copy()
    v0 = z.copy()

    # boundary for first order derivative term
    Bdudx0_cur = np.kron(u0[x0_raveled_indicies], x0_one_hot)
    Bdudy0_cur = np.kron(y0_one_hot, u0[y0_raveled_indicies])
    Bdvdx0_cur = np.kron(v0[x0_raveled_indicies], x0_one_hot)
    Bdvdy0_cur = np.kron(y0_one_hot, v0[y0_raveled_indicies])
    Bdudx1_cur = np.kron(u0[x1_raveled_indicies], x1_one_hot)
    Bdudy1_cur = np.kron(y1_one_hot, u0[y1_raveled_indicies])
    Bdvdx1_cur = np.kron(v0[x1_raveled_indicies], x1_one_hot)
    Bdvdy1_cur = np.kron(y1_one_hot, v0[y1_raveled_indicies])

    # boundary for second order derivative term
    bxu_cur = np.zeros(nxy)
    byu_cur = np.zeros(nxy)
    bxv_cur = np.zeros(nxy)
    byv_cur = np.zeros(nxy)

    bxu_cur[inner_x_raveled_indicies] = u0[x01_raveled_indicies]
    byu_cur[inner_y_raveled_indicies] = u0[y01_raveled_indicies]
    bxv_cur[inner_x_raveled_indicies] = v0[x01_raveled_indicies]
    byv_cur[inner_y_raveled_indicies] = v0[y01_raveled_indicies]

    def r(u_free, v_free, u_free_prev, v_free_prev, Mu_free, Mv_free, Nu_free, Nv_free,
          Bdudx0_cur, Bdvdx0_cur, Bdudx1_cur, Bdvdx1_cur, Bdudy0_cur, Bdvdy0_cur, Bdudy1_cur, Bdvdy1_cur,
          bxu_cur, bxv_cur, byu_cur, byv_cur):

        f_u = (-1 / dx * (u_free * (Mu_free - Bdudx0_cur))
               - 1 / dy * (v_free * (Nu_free - Bdudy0_cur))
               + 1 / (Re * dx ** 2) * (Dx.dot(u_free) + bxu_cur)
               + 1 / (Re * dy ** 2) * (Dy.dot(u_free) + byu_cur))

        f_v = (-1 / dx * (u_free * (Mv_free - Bdvdx0_cur))
               - 1 / dy * (v_free * (Nv_free - Bdvdy0_cur))
               + 1 / (Re * dx ** 2) * (Dx.dot(v_free) + bxv_cur)
               + 1 / (Re * dy ** 2) * (Dy.dot(v_free) + byv_cur))

        r_u = u_free - u_free_prev - dt * f_u
        r_v = v_free - v_free_prev - dt * f_v

        return np.concatenate((r_u, r_v))

    def J(u_free, v_free, Mu_free, Mv_free, Nu_free, Nv_free,
          Bdudx0_cur, Bdvdx0_cur, Bdudx1_cur, Bdvdx1_cur, Bdudy0_cur, Bdvdy0_cur, Bdudy1_cur, Bdvdy1_cur):

        df_udu = (-1 / dx * (sp.diags(Mu_free - Bdudx0_cur, 0, (nxy, nxy), format="csr")
                             + sp.diags(u_free, 0, (nxy, nxy), format="csr").dot(Mb))
                  - 1 / dy * sp.diags(v_free, 0, (nxy, nxy), format="csr").dot(Nb)
                  + 1 / (Re * dx ** 2) * Dx
                  + 1 / (Re * dy ** 2) * Dy)
        df_udv = -1 / dy * sp.diags(Nu_free - Bdudy0_cur, 0, (nxy, nxy), format="csr")
        df_vdu = -1 / dx * sp.diags(Mv_free - Bdvdx0_cur, 0, (nxy, nxy), format="csr")
        df_vdv = (-1 / dx * sp.diags(u_free, 0, (nxy, nxy), format="csr").dot(Mb)
                  - 1 / dy * (sp.diags(Nv_free - Bdvdy0_cur, 0, (nxy, nxy), format="csr")
                              + sp.diags(v_free, 0, (nxy, nxy), format="csr").dot(Nb))
                  + 1 / (Re * dx ** 2) * Dx
                  + 1 / (Re * dy ** 2) * Dy)

        return sp.bmat([[I - dt * df_udu, -dt * df_udv], [-dt * df_vdu, I - dt * df_vdv]], format='csr')

    # solution snapshot
    u_full = np.zeros(((nt + 1), ny * nx))
    v_full = np.zeros(((nt + 1), ny * nx))

    # solution + intermediate snapshot
    u_full_inter = np.array([])
    v_full_inter = np.array([])

    # IC
    u_full[0] = np.copy(u0)
    v_full[0] = np.copy(v0)
    u0_free = u0[free_raveled_indicies]
    v0_free = v0[free_raveled_indicies]

    for k in range(nt):
        u_free_prev = np.copy(u_full[k, free_raveled_indicies])
        v_free_prev = np.copy(v_full[k, free_raveled_indicies])

        u_free = np.copy(u_full[k, free_raveled_indicies])
        v_free = np.copy(v_full[k, free_raveled_indicies])

        Mu_free = Mb.dot(u_free)
        Mv_free = Mb.dot(v_free)
        Nu_free = Nb.dot(u_free)
        Nv_free = Nb.dot(v_free)

        residual = r(u_free, v_free, u_free_prev, v_free_prev, Mu_free, Mv_free, Nu_free, Nv_free,
                     Bdudx0_cur, Bdvdx0_cur, Bdudx1_cur, Bdvdx1_cur, Bdudy0_cur, Bdvdy0_cur, Bdudy1_cur, Bdvdy1_cur,
                     bxu_cur, bxv_cur, byu_cur, byv_cur)

        for itr in range(maxitr):
            jacobian = J(u_free, v_free, Mu_free, Mv_free, Nu_free, Nv_free,
                         Bdudx0_cur, Bdvdx0_cur, Bdudx1_cur, Bdvdx1_cur, Bdudy0_cur, Bdvdy0_cur, Bdudy1_cur, Bdvdy1_cur)

            delta_free = spsolve(jacobian, -residual)

            u_free += delta_free[:nxy]
            v_free += delta_free[nxy:]

            Mu_free = Mb.dot(u_free)
            Mv_free = Mb.dot(v_free)
            Nu_free = Nb.dot(u_free)
            Nv_free = Nb.dot(v_free)

            residual = r(u_free, v_free, u_free_prev, v_free_prev, Mu_free, Mv_free, Nu_free, Nv_free,
                         Bdudx0_cur, Bdvdx0_cur, Bdudx1_cur, Bdvdx1_cur, Bdudy0_cur, Bdvdy0_cur, Bdudy1_cur, Bdvdy1_cur,
                         bxu_cur, bxv_cur, byu_cur, byv_cur)

            # store itermediate values
            R = np.linalg.norm(residual)

            if R < tol:
                u_full[k + 1, free_raveled_indicies] = np.copy(u_free)
                v_full[k + 1, free_raveled_indicies] = np.copy(v_free)
                # BC from exact solution
                u_full[k + 1, fixed_raveled_indicies] = np.copy(u0[fixed_raveled_indicies])
                v_full[k + 1, fixed_raveled_indicies] = np.copy(v0[fixed_raveled_indicies])
                break

        if R >= tol:
            print("\n non converged after {}th iteration".format(maxitr))
            break
    timer.append(time())
    time_fom = timer[1] - timer[0]

    return u_full.reshape(-1, ny * nx), v_full.reshape(-1, ny * nx), time_fom
