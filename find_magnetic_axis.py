import numpy as np

def find_magnetic_axis(biotsavart, n, rguess, output='cylindrical'):
    assert output in ['cylindrical', 'cartesian']
    from scipy.spatial.distance import cdist
    from scipy.optimize import fsolve
    points = np.linspace(0, 2*np.pi, n, endpoint=False).reshape((n, 1))
    oneton = np.asarray(range(0, n)).reshape((n, 1))
    fak = 2*np.pi / (points[-1] - points[0] + (points[1]-points[0]))
    dists = fak * cdist(points, points, lambda a, b: a-b)
    np.fill_diagonal(dists, 1e-10)  # to shut up the warning
    if n % 2 == 0:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.tan(0.5 * dists)
    else:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.sin(0.5 * dists)

    np.fill_diagonal(D, 0)
    D *= fak
    phi = points

    def build_residual(rz):
        inshape = rz.shape
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        B = biotsavart.B()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        Br = np.cos(phi)*Bx + np.sin(phi)*By
        Bphi = np.cos(phi)*By - np.sin(phi)*Bx
        residual_r = D @ r - r * Br / Bphi
        residual_z = D @ z - r * Bz / Bphi
        return np.vstack((residual_r, residual_z)).reshape(inshape)

    def build_jacobian(rz):
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        GradB = biotsavart.dB_by_dX()
        B = biotsavart.B()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        dxBx = GradB[:, 0, 0].reshape((n, 1))
        dyBx = GradB[:, 1, 0].reshape((n, 1))
        dzBx = GradB[:, 2, 0].reshape((n, 1))
        dxBy = GradB[:, 0, 1].reshape((n, 1))
        dyBy = GradB[:, 1, 1].reshape((n, 1))
        dzBy = GradB[:, 2, 1].reshape((n, 1))
        dxBz = GradB[:, 0, 2].reshape((n, 1))
        dyBz = GradB[:, 1, 2].reshape((n, 1))
        dzBz = GradB[:, 2, 2].reshape((n, 1))
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        Br = cosphi*Bx + sinphi*By
        Bphi = cosphi*By - sinphi*Bx
        drBr = cosphi*cosphi * dxBx + cosphi*sinphi*dyBx + sinphi*cosphi*dxBy + sinphi*sinphi*dyBy
        dzBr = cosphi*dzBx + sinphi*dzBy
        drBphi = cosphi*cosphi*dxBy + cosphi*sinphi*dyBy - sinphi*cosphi*dxBx - sinphi*sinphi*dyBx
        dzBphi = cosphi*dzBy - sinphi*dzBx
        drBz = cosphi * dxBz + sinphi*dyBz
        # residual_r = D @ r - r * Br / Bphi
        # residual_z = D @ z - r * Bz / Bphi
        dr_resr = D + np.diag((-Br/Bphi - r*drBr/Bphi + r*Br*drBphi/Bphi**2).reshape((n,)))
        dz_resr = np.diag((-r*dzBr/Bphi + r*Br*dzBphi/Bphi**2).reshape((n,)))
        dr_resz = np.diag((-Bz/Bphi - r*drBz/Bphi + r*Bz*drBphi/Bphi**2).reshape((n,)))
        dz_resz = D + np.diag((-r*dzBz/Bphi + r*Bz*dzBphi/Bphi**2).reshape((n,)))
        return np.block([[dr_resr, dz_resr], [dr_resz, dz_resz]])
    
    r0 = np.ones_like(phi) * rguess
    z0 = np.zeros_like(phi)
    x0 = np.vstack((r0, z0))
    # h = np.random.rand(x0.size).reshape(x0.shape)
    # eps = 1e-4
    # drdh = build_jacobian(x0)@h
    # drdh_est = (build_residual(x0+eps*h)-build_residual(x0-eps*h))/(2*eps)
    # err = np.linalg.norm(drdh-drdh_est)
    # print(err)
    # print(np.hstack((drdh, drdh_est)))

    # diff = 1e10
    # soln = x0.copy()
    # for i in range(50):
        # r = build_residual(soln)
        # print("r", np.linalg.norm(r))
        # update = np.linalg.solve(build_jacobian(soln), r)
        # soln -= 0.01 * update
        # diff = np.linalg.norm(update)
        # print('dx', diff)
    soln = fsolve(build_residual, x0, fprime=build_jacobian, xtol=1e-13)
    if output == 'cylindrical':
        return np.hstack((soln[:n, None], phi, soln[n:, None]))
    else:
        return np.hstack((soln[:n, None]*np.cos(phi), soln[:n, None]*np.sin(phi), soln[n:, None]))

