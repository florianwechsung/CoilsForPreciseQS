#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.curve import curves_to_vtk
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curveobjectives import CurveLength, CoshCurveCurvature
from simsopt.geo.curveobjectives import MinimumDistance
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.surfaceobjectives import QfmResidual, ToroidalFlux, Area, Volume
from objective import create_curves
from scipy.optimize import minimize
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, default=1e-3)
parser.add_argument("--sampleidx", type=int, default=-1)
parser.add_argument("--outdiridx", type=int, default=0)
parser.add_argument("--well", dest="well", default=False, action="store_true")
args = parser.parse_args()

if args.sampleidx == -1:
    sampleidx = None
else:
    sampleidx = args.sampleidx
if not args.well:
    filename = 'input.LandremanPaul2021_QA'
    outdirs = [
        "output/well_False_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_expquad/",
        "output/well_False_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad/",
        "output/well_False_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_4_order_16_expquad/",
        "output/well_False_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_2_order_16_expquad/",
    ]
else:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"
    outdirs = [
        "output/well_True_lengthbound_18.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad/",
        "output/well_True_lengthbound_20.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_4_order_16_expquad/",
        "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_7_order_16_expquad/",
        "output/well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_5_order_16_expquad/",
    ]

fil = 0
nfp = 2

nphi = 25
ntheta = 25
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
phis += phis[0]/2
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
sigma = args.sigma
outdir = outdirs[args.outdiridx]
x = np.loadtxt(outdir + "xmin.txt")

nsamples = 0 if sampleidx is None else sampleidx + 1
base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=fil, ig=0, nsamples=nsamples, stoch_seed=1, sigma=sigma, order=16)

if sampleidx is None:
    coils_qfm = coils_fil
else:
    coils_qfm = coils_fil_pert[sampleidx]
mpol = s.mpol
ntor = s.ntor

sq = SurfaceRZFourier(mpol=8, ntor=8, nfp=2, stellsym=True, quadpoints_phi=phis, quadpoints_theta=thetas)
for m in range(0, 6):
    for n in range(-5, 6):
        sq.set_rc(m, n, s.get_rc(m, n))
        sq.set_zs(m, n, s.get_zs(m, n))

phisfine = np.linspace(0, 1./(2*nfp), 2*nphi, endpoint=False)
phisfine += phisfine[0]/2
thetasfine = np.linspace(0, 1., 2*ntheta, endpoint=False)
sqfine = SurfaceRZFourier(mpol=16, ntor=16, nfp=2, stellsym=True, quadpoints_phi=phisfine, quadpoints_theta=thetasfine)
def transfer(sq, sqfine):
    for m in range(0, 9):
        for n in range(-8, 9):
            sqfine.set_rc(m, n, sq.get_rc(m, n))
            sqfine.set_zs(m, n, sq.get_zs(m, n))


print(len(sq.get_dofs()), "vs", nphi*ntheta)
bs = BiotSavart(coils_qfm)
bs.x = x
from find_magnetic_axis import find_magnetic_axis
axis = find_magnetic_axis(bs, 200, 1.0, output='cartesian')
from simsopt.geo.curverzfourier import CurveRZFourier 
ma = CurveRZFourier(200, 10, 1, False)
ma.least_squares_fit(axis)
curves_to_vtk([ma], "/tmp/axis")

if sampleidx is None:
    outname = outdir + f"qfm_{sampleidx}"
else:
    outname = outdir + f"qfm_seed_{sampleidx}_sigma_{sigma}"
bs_tf = BiotSavart(coils_qfm)
bs_tf.x = x
tf = ToroidalFlux(sq, bs_tf)
tf_init = tf.J()
bs_tffine = BiotSavart(coils_qfm)
bs_tffine.x = x
tffine = ToroidalFlux(sqfine, bs_tffine)
# tf_init = tf.J()
# print(tf.J())
# sq.extend_via_normal(-0.04)
# print(tf.J())
# sq.to_vtk("/tmp/shrunk")

faks = [1.0, 0.25]
for i, f in enumerate(faks):
    sq.to_vtk(f"/tmp/pre_{f}")
    tf_target = tf_init * f
    qfm = QfmResidual(sq, bs)
    qfm_surface = QfmSurface(bs, sq, tf, tf_target)

    constraint_weight = 1
    print("intial qfm value", qfm.J())

    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-14, maxiter=1600,
                                                             constraint_weight=constraint_weight)
    print(f"||ar constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    transfer(sq, sqfine)
    qfm = QfmResidual(sqfine, bs)
    qfm_surface = QfmSurface(bs, sqfine, tffine, tf_target)

    # constraint_weight = 1
    print("intial qfm value", qfm.J())
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-18, maxiter=1600,
                                                             constraint_weight=constraint_weight)
    print(f"||ar constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-20, maxiter=1600,
                                                             constraint_weight=constraint_weight)
    print(f"||ar constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

    print("Volume", sqfine.volume())
    print("Area", sqfine.area())
    np.save(outname + f"_flux_{f}", sqfine.get_dofs())
    np.save("outputforvmec/" + outname.replace("/", "_") + f"_flux_{f}", sqfine.get_dofs())


    sq.to_vtk(f"/tmp/opt_{f}")
    if i < len(faks)-1:
        print("Before scale", tf.J())
        sq.scale_around_curve(ma, (faks[i+1]/faks[i])**0.5)
        print("After scale", tf.J())
