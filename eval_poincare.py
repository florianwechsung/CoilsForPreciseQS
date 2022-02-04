#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curve import curves_to_vtk
from objective import create_curves
import numpy as np
from simsopt.field.tracing import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
import os
os.makedirs("poincare", exist_ok=True)
from mpi4py import MPI
comm = MPI.COMM_WORLD

filename = 'input.LandremanPaul2021_QA'
outdirs = [
    # "output/well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_expquad/",
    # "output/well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad/",
    # "output/well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_4_order_16_expquad/",
    "output/well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_expquad/",
]

# filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"
# outdirs = [
#     "output/well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad/",
#     "output/well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_4_order_16_expquad/",
#     "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_expquad/",
#     "output/well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_expquad/",
# ]

extend = True

fil = 0
nfp = 2
noutsamples = 256
nphi = 128
ntheta = 64
phis = np.linspace(0, 1., nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
s2 = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
s2.extend_via_normal(0.01)
if extend:
    s2.scale(1.5)
    s2.extend_via_normal(0.3)
sc_fieldline = SurfaceClassifier(s2, h=0.05, p=3)
s2.to_vtk("/tmp/s2")
s.to_vtk("/tmp/s")

fluxs = []
for idx in range(len(outdirs)):
    outdir = outdirs[idx]
    try:
        x = np.loadtxt(outdir + "xmin.txt")
    except:
        print("skipping", outdir)
        continue

    base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
        fil=fil, ig=0, nsamples=0, stoch_seed=0, sigma=0, zero_mean=False, order=16)

    bs = BiotSavart(coils_fil)
    bs.x = x
    meanb = np.mean(bs.set_points(s.gamma().reshape((-1, 3))).AbsB())
    bs = (1./meanb) * bs

    n = 60
    degree = 5
    rs = np.linalg.norm(s2.gamma()[:, :, 0:2], axis=2)
    zs = s2.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2*np.pi/2, n*2)
    zrange = (0, np.max(zs), n//2)
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=2, stellsym=True
    )

    bsh.set_points(s.gamma().reshape((-1, 3)))
    dBh = bsh.GradAbsB()
    Bh = bsh.B()
    bs.set_points(s.gamma().reshape((-1, 3)))
    dB = bs.GradAbsB()
    B = bs.B()
    print("B    errors on surface %s" % np.sort(np.abs(B-Bh).flatten()), flush=True)
    print("∇|B| errors on surface %s" % np.sort(np.abs(dB-dBh).flatten()), flush=True)

    nfieldlines = 12
    if extend:
        nfieldlines += 21
        if 'well' in filename:
            R0 = np.linspace(1.2535459, 1.45, nfieldlines)
        else:
            R0 = np.linspace(1.2124936, 1.45, nfieldlines)
    else:
        if 'well' in filename:
            R0 = np.linspace(1.2535459, s.gamma()[0, 0, 0], nfieldlines)
        else:
            R0 = np.linspace(1.2124936, s.gamma()[0, 0, 0], nfieldlines)

    Z0 = [0. for i in range(nfieldlines)]
    phis = [(j/4)*(2*np.pi/nfp) for j in range(4)]
    # tmax_fl = 1000
    print("This will take a while. Consider reducing `tmax_fl` to compute a lower resolution poincare plot.")
    # tmax_fl = 70000
    tmax_fl = 60000
    tol = 1e-17

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bsh, R0, Z0, tmax=tmax_fl, tol=tol, comm=comm,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)],
        forget_exact_path=True)

    if comm is None or comm.rank == 0:
        for phiidx in range(4):
            data_out = []
            labels_out = []
            for i in range(nfieldlines):
                xyz = fieldlines_phi_hits[i][fieldlines_phi_hits[i][:, 1] == phiidx, 2:]
                r = np.linalg.norm(xyz[:, :2], axis=1)
                z = xyz[:, 2]
                data_out.extend([r, z])
                labels_out.extend([f"r_{i}", f"z_{i}"])
            cross_section = s.cross_section(phis[phiidx], thetas=np.linspace(0, 1, 100, endpoint=True))
            data_out.extend([np.linalg.norm(cross_section[:, :2], axis=1), cross_section[:, 2]])
            labels_out.extend([f"r_surf", f"z_surf"])
            import pandas as pd
            data_out = pd.DataFrame(data_out).to_numpy()
            outname = outdir + f"_poincare_{phiidx}"
            np.savetxt(f"poincare/" + outname.replace(".", "_").replace("/", "_") + ".txt", data_out.T, comments='', delimiter=',', header=",".join(labels_out))

        # particles_to_vtk(fieldlines_tys, f'/tmp/fieldlines_{idx}')
        plot_poincare_data(fieldlines_phi_hits, phis, f'/tmp/poincare_fieldline_{idx}.png', dpi=250)

