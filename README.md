# Coils for precise Quasi Symmetry

![QA+Well[24] coils](https://raw.githubusercontent.com/florianwechsung/CoilsForPreciseQS/main/coils_well24.png)

This repository contains code to accompany the manuscript

- *Precise stellarator quasi-symmetry can be achieved with electromagnetic coils*, Florian Wechsung, Matt Landreman , Andrew Giuliani, Antoine Cerfon, and Georg Stadler

The goal is to compute coils that approximate the QA and the QA+Well magnetic field configurations found in 

- *Magnetic fields with precise quasisymmetry for plasma confinement.* Landreman, Matt, and Elizabeth Paul. Physical Review Letters 128.3 (2022): 035001.


## Installation

To use this code, first clone the repository including all its submodules, via

    git clone --recursive 

and then install [SIMSOPT](https://github.com/hiddenSymmetries/simsopt) via

    pip install -e simsopt/

If you have any trouble with the installation of SIMSOPT, please refer to the installation instructions [here](https://simsopt.readthedocs.io/en/latest/installation.html#virtual-environments) or open an [issue](https://github.com/hiddenSymmetries/simsopt/issues).

## Basic Usage

Once you have installed SIMSOPT, you can run

    python3 driver.py --order 16 --lengthbound 18 --mindist 0.10 --maxkappa 5 --maxmsc 5  --ig 0 --expquad

to find coils that approximate the **QA** configuration in Landreman \& Paul. The configuration you obtain from this command corresponds to the `QA[18]` configuration in the Precise QS paper.

Here the options mean:

- `order`: at what order to truncate the Fourier series for the curves that represent the coils.
- `lengthbound`: the maximum total length of the four modular coils.
- `mindist`: the minimum distance between coils to be enforced.
- `maxkappa`: the maximum allowable curvature of the coils.
- `maxmsc`: the maximum allowable mean squared curvature of the coils.
- `ig`: which initial guess to choose. `0` corresponds to flat coils, other values result in random perturbations of the flat coils. In the paper we picked `ig\in\{0,...,7\}`
- `expquad`: turns on a modified quadrature scheme that results in exponential
  convergence of the surface integral in the objective(recommended).

If you would like to target the QA+Well configuration instead, simply add `--well` to the command.

These optimizations take a while on a small machine, so instead, you can also just unzip the `archive.zip` file, which will restore the output from the runs used in the paper.

## Analysis

Since the optimization routine may find a different minimizer depending on the initial guess, we first run

    python3 eval_find_best.py  --order 16 --lengthbound 18 --mindist 0.10 --maxkappa 5 --maxmsc 5 --expquad

to find the best of the local minimizers. Once this is done, we can compute basic properties of the coils and the magnetic field.

First run

    python3 eval_geo.py

to print geometric properties of the coils. Then run

    python3 eval_vis.py

to create some paraview files that visualize both the coils and field. To compute the Poincare plot in the paper, run

    python3 eval_poincare.py
    python3 eval_poincare_vis.py

The computation of the symmetry-breaking Fourier modes and the alpha particle confinement are done via BOOZXFORM and SIMPLE. These require a VMEC solution as input. To obtain a VMEC solution, we compute a magnetic surface of the Biot-Savart induced field and pass that to VMEC. In order to do this, simply run

    python3 eval_compute_multiple_qfm.py --outdiridx 0

and increase `outdiridx` to select the configuration with coil length 18, 20, 22, or 24. Add `--well` for the QA+Well configuration.
