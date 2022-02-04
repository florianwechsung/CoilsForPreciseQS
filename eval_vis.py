#!/usr/bin/env python
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.biotsavart import BiotSavart
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
from simsopt.geo.curve import curves_to_vtk
from objective import create_curves
import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--well", dest="well", default=False, action="store_true")
args = parser.parse_args()

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
        # "output/well_True_lengthbound_22.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad_samples_4096_sigma_0.001_usedetig_dashfix/",
        # "output/temp_well_True_lengthbound_24.0_kap_5.0_msc_5.0_dist_0.1_fil_0_ig_6_order_16_expquad_samples_4096_sigma_0.001_usedetig_dashfix/",
    ]

fil = 0

nfp = 2
nphi = 64
ntheta = 64
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
thetas = np.linspace(0, 1., ntheta, endpoint=False)
starget = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)
area_targets = np.flipud(np.linspace(2, starget.area(), 10, endpoint=True))

phisvizfull = np.linspace(0, 1., 1024, endpoint=True)
thetasvizfull = np.linspace(0, 1., 128, endpoint=True)
svizfull = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phisvizfull, quadpoints_theta=thetasvizfull)

nfp = 2
ntheta = 64
nphi = 64
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
phis += phis[1]/2
thetas = np.linspace(0, 1., ntheta, endpoint=False)
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)

base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=fil, ig=0, nsamples=0, stoch_seed=0, sigma=0, order=16)

def coilpy_plot(curves, filename, height=0.1, width=0.1):
    def wrap(data):
        return np.concatenate([data, [data[0]]])
    xx = [wrap(c.gamma()[:, 0]) for c in curves]
    yy = [wrap(c.gamma()[:, 1]) for c in curves]
    zz = [wrap(c.gamma()[:, 2]) for c in curves]
    II = [1. for _ in curves]
    names = [i for i in range(len(curves))]
    from coilpy import Coil
    coils = Coil(xx, yy, zz, II, names, names)
    coils.toVTK(filename, line=False, height=height, width=width)


for outdir in outdirs:
    x = np.loadtxt(outdir + "xmin.txt")
    coils_boozer = coils_fil
    bs = BiotSavart(coils_boozer)
    bs.x = x
    B_on_surface = bs.set_points(s.gamma().reshape((-1, 3))).AbsB()
    norm = np.linalg.norm(s.normal().reshape((-1, 3)), axis=1)
    meanb = np.mean(B_on_surface * norm)/np.mean(norm)

    coilpy_plot([c.curve for c in coils_fil], outdir + "coils_fb.vtu", height=0.05, width=0.05)
    pointData = {
        "B·n/|B|": np.sum(bs.set_points(svizfull.gamma().reshape((-1, 3))).B().reshape(svizfull.gamma().shape) * svizfull.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape(svizfull.gamma().shape[:2] + (1,)),
        "|B|": bs.AbsB().reshape(svizfull.gamma().shape[:2] + (1,))/meanb
    }
    svizfull.to_vtk(outdir + "surf_opt_vis_full", extra_data=pointData)
    print(f"Created coils_fb.vtu and surf_opt_vis_full.vts in directory {outdir}")

