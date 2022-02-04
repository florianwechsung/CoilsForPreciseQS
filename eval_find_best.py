from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.curve import curves_to_vtk
from objective import create_curves, MeanSquareCurvature
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.)
parser.add_argument("--fil", type=int, default=0)
parser.add_argument("--nsamples", type=int, default=0)
parser.add_argument("--order", type=int, default=20)
parser.add_argument("--sigma", type=float, default=0.001)
parser.add_argument("--lengthbound", type=float, default=0.)
parser.add_argument("--noutsamples", type=int, default=0)
parser.add_argument("--mindist", type=float, default=0.10)
parser.add_argument("--maxkappa", type=float, default=5.0)
parser.add_argument("--well", dest="well", default=False, action="store_true")
parser.add_argument("--zeromean", dest="zeromean", default=False, action="store_true")
parser.add_argument("--usedetig", dest="usedetig", default=False, action="store_true")
parser.add_argument("--noalen", dest="noalen", default=False, action="store_true")
parser.add_argument("--maxmsc", type=float, default=5)
parser.add_argument("--expquad", dest="expquad", default=False, action="store_true")
parser.add_argument("--glob", dest="glob", default=False, action="store_true")
parser.add_argument("--fixcurrents", dest="fixcurrents", default=False, action="store_true")
parser.add_argument("--hybrid", dest="hybrid", default=False, action="store_true")
args = parser.parse_args()

if args.nsamples == 0:
    args.sigma = 0.

nfp = 2
# nphi = 32
# ntheta = 32
nphi = 64
ntheta = 64
phis = np.linspace(0, 1./(2*nfp), nphi, endpoint=False)
phis += phis[1]/2


nphi = 128
ntheta = 32
phis = np.linspace(0, 1., nphi, endpoint=False)
phis += phis[1]/2


thetas = np.linspace(0, 1., ntheta, endpoint=False)
if args.well:
    filename = "input.20210728-01-010_QA_nfp2_A6_magwell_weight_1.00e+01_rel_step_3.00e-06_centered"
else:
    filename = "input.LandremanPaul2021_QA"
s = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phis, quadpoints_theta=thetas)

phisviz = np.linspace(0, 1./(2*nfp), 2*nphi, endpoint=True)
thetasviz = np.linspace(0, 1., 2*ntheta, endpoint=True)
sviz = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phisviz, quadpoints_theta=thetasviz)
phisvizfull = np.linspace(0, 1., 8*nphi, endpoint=True)
thetasvizfull = np.linspace(0, 1., 2*ntheta, endpoint=True)
svizfull = SurfaceRZFourier.from_vmec_input(filename, quadpoints_phi=phisvizfull, quadpoints_theta=thetasvizfull)



def getoutdir(ig):
    outdir = f"output/well_{args.well}_lengthbound_{args.lengthbound}_kap_{args.maxkappa}_msc_{args.maxmsc}_dist_{args.mindist}_fil_{args.fil}_ig_{ig}_order_{args.order}"
    if args.noalen:
        outdir += "_noalen"
    if args.expquad:
        outdir += "_expquad"
    if args.glob:
        outdir += "_nonlocal"
    if args.nsamples > 0:
        outdir += f"_samples_{args.nsamples}_sigma_{args.sigma}"
        if args.zeromean:
            outdir += f"_zeromean_{args.zeromean}"
        if args.usedetig:
            outdir += "_usedetig"
        if args.fixcurrents:
            outdir += "_fixcurrents"
        if args.hybrid:
            outdir += "_hybrid"
        outdir += "_dashfix"
    outdir += "/"
    return outdir


base_curves, base_currents, coils_fil, coils_fil_pert = create_curves(
    fil=args.fil, ig=0, nsamples=args.nsamples,
    stoch_seed=0, sigma=args.sigma,
    zero_mean=args.zeromean, order=args.order)


bestval = 1e10
bestidx = -1
for ig in range(8):
    outdir = getoutdir(ig)
    try:
        x = np.loadtxt(outdir + "xmin.txt")
        grad = np.loadtxt(outdir + "grad.txt")
    except:
        print("skipping", outdir)
        continue
    print(f"IG={ig}")
    bs = BiotSavart(coils_fil)
    bs.x = x
    bs.set_points(s.gamma().reshape((-1, 3)))
    # for i in range(4):
    #     print(MeanSquareCurvature(coils_fil[i].curve, 0.).msc())
    detval = SquaredFlux(s, bs, local=(not args.glob)).J()
    print("Det", detval)
    print("grad", np.linalg.norm(grad))

    pointData = {"B_N/|B|": np.sum(bs.set_points(sviz.gamma().reshape((-1, 3))).B().reshape(sviz.gamma().shape) * sviz.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape((2*nphi, 2*ntheta, 1))}
    sviz.to_vtk(outdir + "surf_opt_vis", extra_data=pointData)
    pointData = {"B_N/|B|": np.sum(bs.set_points(svizfull.gamma().reshape((-1, 3))).B().reshape(svizfull.gamma().shape) * svizfull.unitnormal(), axis=2)[:, :, None]/bs.AbsB().reshape((8*nphi, 2*ntheta, 1))}
    svizfull.to_vtk(outdir + "surf_opt_vis_full", extra_data=pointData)
    curves_to_vtk([c.curve for c in coils_fil], outdir + "coil_opt_vis", close=True)

    stochval = 0
    for coils in coils_fil_pert:
        bs = BiotSavart(coils)
        bs.x = x
        bs.set_points(s.gamma().reshape((-1, 3)))
        stochval += SquaredFlux(s, bs, local=(not args.glob)).J()/args.nsamples
        errs = []
        for i in range(len(coils_fil)):
            errs.append(np.max(np.linalg.norm(coils_fil[i].curve.gamma()-coils[i].curve.gamma(), axis=1)))
        # print(errs)
    if args.nsamples > 0:
        print("Insample", stochval)
        if stochval < bestval:
            bestidx = ig
            bestval = stochval
    else:
        if detval < bestval:
            bestidx = ig
            bestval = detval

print("Best outdir")
print(getoutdir(bestidx))
print("Best val", bestval)
