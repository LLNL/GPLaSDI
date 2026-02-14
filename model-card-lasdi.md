---
language:
- en # ISO language tag
tags:
- project:genesis # include on all GENESIS project models
- project:LaSDI # include your _short_ model team name e.g. MOAT
- type:model # use other types include {agent, eval, framework, model, etc...}
- science:computational # what kind of science is this for (e.g., materials, biology, lightsource, fusion, climate, etc.)
- risk:general # indicates level of risk review {general, reviewed, restricted}
license: MIT
datasets:
    - # a list of download URLs for dataset files used for training, mid-training, post-training, etc...
metrics:
    - relative_error
    - reconstruction_loss
    - integration_loss
    - speedup_factor
    - training_loss
---

# LaSDI

LaSDI (Latent Space Dynamics Identification) is a family of data-driven reduced-order modeling frameworks for fast parametric physical simulations, developed at Lawrence Livermore National Laboratory and collaborating institutions.

*Last Updated*: **2025-07-15**

## Developed by

- William D. Fries (University of Arizona) — LaSDI
- Christophe Bonneville (Cornell University) — GPLaSDI
- April Tran (University of Colorado Boulder) — WLaSDI
- Jun Sur R. Park (Korea Institute for Advanced Study) — tLaSDI
- Seung Whan Chung (Lawrence Livermore National Laboratory) — LaSDI-IT
- William Anderson (Lawrence Livermore National Laboratory) — mLaSDI, HLaSDI
- Robert Stephany (Lawrence Livermore National Laboratory) — mLaSDI, HLaSDI
- Youngsoo Choi (Lawrence Livermore National Laboratory) — all variants

## Contributed by

- Xiaolong He (UC San Diego / ANSYS Inc.) — LaSDI, WLaSDI
- Debojyoti Ghosh (LLNL) — GPLaSDI
- Jonathan L. Belof (LLNL) — GPLaSDI
- Daniel A. Messenger (University of Colorado Boulder) — WLaSDI
- David M. Bortz (University of Colorado Boulder) — WLaSDI
- Siu Wun Cheung (LLNL) — tLaSDI
- Yeonjong Shin (North Carolina State University / POSTECH) — tLaSDI
- Christopher Miller (LLNL) — LaSDI-IT
- Paul Tranquilli (LLNL) — LaSDI-IT
- H. Keo Springer (LLNL) — LaSDI-IT
- Kyle Sullivan (LLNL) — LaSDI-IT 

## Model Changelog

+ **2022-08-10** LaSDI: Parametric Latent Space Dynamics Identification
+ **2024-01-01** GPLaSDI: Gaussian Process-based LaSDI
+ **2024-03-09** tLaSDI: Thermodynamics-informed LaSDI
+ **2024-04-01** WLaSDI: Weak-form Latent Space Dynamics Identification
+ **2025-07-15** LaSDI-IT: LaSDI for Interface Tracking
+ **2025-12-17** HLaSDI: Higher-Order LaSDI
+ **2025-12-23** mLaSDI: Multi-stage LaSDI

## Model short description

Data-driven reduced-order modeling framework that learns latent space dynamics for fast and accurate parametric physical simulations. 

## Model description

1. LaSDI compresses high-dimensional PDE solution data into a low-dimensional latent space using either POD (linear) or autoencoders (nonlinear).
2. Governing dynamics in the latent space are identified via system identification techniques such as SINDy, Gaussian Process regression, WENDy (weak-form), or GFINNs (thermodynamics-informed).
3. The identified latent dynamics are solved for new parameter values and reconstructed to the full state, achieving O(100)x to O(10^6)x speedup with O(1)% relative error.
4. Variants include: GPLaSDI (greedy sampling with uncertainty quantification), WLaSDI (noise-robust weak formulation), tLaSDI (thermodynamic structure preservation), LaSDI-IT (sharp interface tracking), mLaSDI (multi-stage residual learning for high-frequency content), and HLaSDI (higher-order time derivatives).

## Related models

- **LaSDI** (original) — foundation framework
- **GPLaSDI** — extends LaSDI with Gaussian Process interpolation and greedy sampling
- **WLaSDI** — replaces SINDy with WENDy (weak-form) for noise robustness
- **tLaSDI** — replaces SINDy with GFINNs for thermodynamic structure preservation
- **LaSDI-IT** — extends GPLaSDI with interface-aware autoencoder for sharp discontinuities
- **mLaSDI** — extends any LaSDI variant with multi-stage residual decoders for high-frequency recovery
- **HLaSDI** — extends GPLaSDI to PDEs with arbitrary-order time derivatives using K autoencoders

## Model Type

**Compression:**
- POD / SVD (linear subspace) — LaSDI-LS, WLaSDI-LS
- Shallow masked autoencoder (nonlinear manifold) — LaSDI-NM, WLaSDI-NM
- Deep autoencoder — GPLaSDI
- Interface-aware autoencoder — LaSDI-IT
- Hyper-autoencoder (parametric) — tLaSDI
- Multi-stage autoencoder with residual decoders — mLaSDI
- K coupled autoencoders (one per time derivative order) — HLaSDI

**Latent dynamics identification:**
- SINDy (sparse identification of nonlinear dynamics) — LaSDI, WLaSDI
- WENDy (weak-form estimation of nonlinear dynamics) — WLaSDI
- Gaussian Process regression — GPLaSDI, LaSDI-IT
- GFINNs (GENERIC formalism informed neural networks) — tLaSDI
- Higher-order linear ODE system (K-th order) — HLaSDI
 
## Inputs and outputs

**Input:** Parametric time-dependent PDE solution snapshots (time series of full-state field data, e.g., velocity, temperature, density fields). Parameters typically affect initial conditions or physical coefficients.

**Output:** Predicted full-state field solutions at new parameter values, with orders-of-magnitude speedup over full-order simulations. GPLaSDI additionally provides uncertainty estimates via Gaussian Process confidence intervals. 

## Compute Infrastructure

Training and inference are performed on standard computing resources. Problem-specific compute requirements depend on the full-order model (FOM) used to generate training data.

### Hardware

Hardware requirements are modest — LaSDI variants have been demonstrated on standard CPU workstations and GPU-accelerated systems. The FOM data generation step may require HPC resources depending on the application.

### Software

- Python, PyTorch
- LaSDI: https://github.com/LLNL/LaSDI
- GPLaSDI: https://github.com/LLNL/GPLaSDI
- tLaSDI: https://github.com/pjss1223/tLaSDI
- PyWLaSDI: https://github.com/MathBioCU/PyWLaSDI

## Papers and Scientific Outputs

```bibtex
@article{fries2022lasdi,
  title={LaSDI: Parametric Latent Space Dynamics Identification},
  author={Fries, William D. and He, Xiaolong and Choi, Youngsoo},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={399},
  pages={115436},
  year={2022},
  doi={10.1016/j.cma.2022.115436}
}

@article{bonneville2024gplasdi,
  title={GPLaSDI: Gaussian Process-based interpretable Latent Space Dynamics Identification through deep autoencoder},
  author={Bonneville, Christophe and Choi, Youngsoo and Ghosh, Debojyoti and Belof, Jonathan L.},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={418},
  pages={116535},
  year={2024},
  doi={10.1016/j.cma.2023.116535}
}

@article{tran2024wlasdi,
  title={Weak-form latent space dynamics identification},
  author={Tran, April and He, Xiaolong and Messenger, Daniel A. and Choi, Youngsoo and Bortz, David M.},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={427},
  pages={116998},
  year={2024},
  doi={10.1016/j.cma.2024.116998}
}

@article{park2024tlasdi,
  title={tLaSDI: Thermodynamics-informed latent space dynamics identification},
  author={Park, Jun Sur R. and Cheung, Siu Wun and Choi, Youngsoo and Shin, Yeonjong},
  year={2024},
  url={https://arxiv.org/abs/2403.05848}
}

@article{chung2025lasdiit,
  title={Latent Space Dynamics Identification for Interface Tracking with Application to Shock-Induced Pore Collapse},
  author={Chung, Seung Whan and Miller, Christopher and Choi, Youngsoo and Tranquilli, Paul and Springer, H. Keo and Sullivan, Kyle},
  year={2025},
  url={https://arxiv.org/abs/2507.10647}
}

@article{anderson2025mlasdi,
  title={mLaSDI: Multi-stage latent space dynamics identification},
  author={Anderson, William and Chung, Seung Whan and Stephany, Robert and Choi, Youngsoo},
  year={2025},
  url={https://arxiv.org/abs/2506.09207}
}

@article{stephany2025hlasdi,
  title={Higher-Order LaSDI: Reduced Order Modeling with Multiple Time Derivatives},
  author={Stephany, Robert and Anderson, William and Choi, Youngsoo},
  year={2025},
  url={https://arxiv.org/abs/2512.15997}
}
```

## Model License

MIT License

## Contact Info and Model Card Authors

- Seung Whan Chung (chung28@llnl.gov)
- Youngsoo Choi (choi15@llnl.gov)


# Intended Uses

## Intended Use

Accelerating parametric physical simulations where many forward solves are required, such as inverse problems, design optimization, uncertainty quantification, and parameter studies.

### Primary Intended Users

Researchers and scientists working with parametric PDE simulations who need fast surrogate models with quantifiable accuracy.

### Mission Relevance

Supports DOE computational science missions by enabling real-time or near-real-time predictions for applications in materials science, fusion, climate, fluid dynamics, and high-explosive safety analysis.

## Out-of-Scope Use Cases

- Systems without underlying parametric PDE structure
- Problems requiring strict conservation guarantees (unless using tLaSDI for thermodynamic systems)
- Extrapolation far beyond the training parameter range (except tLaSDI which demonstrates limited extrapolation capability)


# How to use

## Install Instructions

Each variant has its own repository:
```bash
# LaSDI
git clone https://github.com/LLNL/LaSDI
# GPLaSDI
git clone https://github.com/LLNL/GPLaSDI
# tLaSDI
git clone https://github.com/pjss1223/tLaSDI
# WLaSDI
git clone https://github.com/MathBioCU/PyWLaSDI
```
See each repository's README for dependency installation.

## Training configuration

Training requires full-order model (FOM) snapshot data at sampled parameter values. Key configuration includes: latent space dimension, candidate library for dynamics identification, and loss function weights. GPLaSDI and LaSDI-IT additionally configure greedy sampling parameters.

## Inference configuration

Given a new parameter value, the trained model encodes the initial condition, solves the identified latent dynamics ODE, and decodes back to the full state. No FOM access is needed at inference time. 
   

# Code snippets of how to use the model

See the example scripts in each repository:
- LaSDI: https://github.com/LLNL/LaSDI/tree/main/examples
- GPLaSDI: https://github.com/LLNL/GPLaSDI/tree/main/examples
- tLaSDI: https://github.com/pjss1223/tLaSDI
- PyWLaSDI: https://github.com/MathBioCU/PyWLaSDI 


# Limitations

## Risks

LaSDI is a scientific computing framework for accelerating parametric PDE simulations. It does not generate text, code, or agentic outputs. It does not pose risks related to cyberattacks, CBRNE weapons development, or novel security vulnerabilities as defined in America's AI Action Plan. The primary risk is inaccurate physical predictions if the model is applied outside its trained parameter range or to systems that violate its underlying assumptions.

## Limitations

- Accuracy degrades for parameter values far from training data (except tLaSDI which offers limited extrapolation)
- Linear compression (POD) is insufficient for advection-dominated problems with slow Kolmogorov n-width decay; nonlinear compression (autoencoder) is needed
- Higher-degree polynomial dynamics can be numerically unstable
- Autoencoder training adds computational overhead to the offline phase


# Training details 

## Training data

**LaSDI:** Four PDE benchmark problems — 1D Burgers (1001 spatial DOF), 2D Burgers, radial advection (9216 spatial DOF), and nonlinear heat conduction (8192 triangular elements). Training snapshots collected on a predefined uniform grid of parameter values. POD or autoencoder used for compression with 3-5 latent dimensions.

**GPLaSDI:** 1D Burgers, 2D Burgers, 1D-1V Vlasov equation (plasma physics), and 2D rising thermal bubble (compressible Euler). Training starts with a few parameter points and adaptively adds new ones via greedy sampling based on GP prediction uncertainty.

**WLaSDI:** Same four problems as LaSDI — 1D inviscid Burgers, 2D viscous Burgers, heat conduction, and radial advection. Training parameters sampled uniformly from the parameter space. Tested with up to 100% Gaussian white noise added to training data.

**tLaSDI:** Couette flow of an Oldroyd-B fluid (400 DOF from 100 mesh nodes with 4 state variables) and 1D parametric inviscid Burgers (201 DOF). 25 training parameter points selected via greedy sampling during training.

**LaSDI-IT:** Shock-induced pore collapse in high explosives with varying pore geometry parameters. Training data generated from high-fidelity ALE3D simulations over an 11-by-11 grid of parameter cases that determine the initial pore shape.

**mLaSDI:** Multiscale oscillating system, unsteady wake flow (incompressible Navier-Stokes, 11,492 mesh nodes), and 1D-1V Vlasov equation. Uses GPLaSDI-style greedy sampling. Source code based on GPLaSDI.

**HLaSDI:** 1D Burgers, wave equation, telegrapher's equation, and Klein-Gordon equation. Training on an 11x11 uniform grid across a 2D parameter space (121 total simulations), with greedy sampling starting from the four corners.

## Training Procedure

All LaSDI variants follow a common procedure: (1) collect FOM snapshots, (2) compress to latent space, (3) identify latent dynamics, (4) predict at new parameters. See the papers and code repositories listed in the Papers and Software sections for full implementation details.

**LaSDI:** Sequential training — compress via POD or autoencoder, then fit polynomial dynamics (up to degree 5) via least-squares regression.

**GPLaSDI:** Simultaneous autoencoder and SINDy training. GP interpolation of ODE coefficients. Greedy sampling adds training points based on prediction uncertainty.

**WLaSDI:** Same compression as LaSDI. Dynamics identified via WENDy (weak-form), which integrates the ODE residual against test functions for noise robustness.

**tLaSDI:** Simultaneous autoencoder and GFINNs training. Enforces thermodynamic structure (energy/entropy conservation) in latent space. Adam optimizer.

**LaSDI-IT:** Autoencoder jointly reconstructs field and material indicator. Linear ODE dynamics to avoid overfitting. GP interpolation with greedy sampling. 

**mLaSDI:** Sequential multi-stage training. First stage trains a standard GPLaSDI autoencoder. Subsequent stages train additional decoders that map the same latent trajectories to residuals from previous stages, using periodic activation functions for high-frequency recovery.

**HLaSDI:** Uses K autoencoders for a K-th order PDE system. Novel loss functions (Consistency, Chain-Rule, IC-Rollout) encode the mathematical structure. GP interpolation with greedy sampling for parametric generalization.

# Evaluation details 

## Evaluation data

Evaluation is performed on unseen parameter values not included in the training set. For each variant, the predicted full-state solution is compared against the corresponding FOM solution. See the papers for specific evaluation parameter grids and test configurations.  

## Evaluation Procedure

The primary evaluation metric is the relative L2 error between the ROM prediction and the FOM solution across the full spatiotemporal domain. Speedup factor is measured as the ratio of FOM wall-clock time to ROM prediction time. Baselines include the full-order model and, where applicable, comparisons between LaSDI variants (e.g., tLaSDI vs gLaSDI, WLaSDI vs LaSDI under noise). 

## Uncertainty Quantification

GPLaSDI, LaSDI-IT, mLaSDI and HLaSDI provide uncertainty quantification through Gaussian Process predictive distributions. The GP confidence intervals on the ODE coefficients propagate through the latent dynamics solver to produce uncertainty bounds on the full-state predictions. Other variants (LaSDI, WLaSDI, tLaSDI) do not include built-in UQ. 

## Evaluation results

**LaSDI:** O(100)x to ~800x speedup with 1-3% relative error across 1D/2D Burgers, radial advection, and heat conduction problems.

**GPLaSDI:** 200x to 100,000x speedup with up to 7% relative error across Burgers, Vlasov, and rising thermal bubble problems.

**WLaSDI:** 140x speedup on 1D Burgers. Under 100% Gaussian noise, maintains <6% error where LaSDI exceeds 10,000% error.

**tLaSDI:** Robust extrapolation beyond training time window with worst-case 1.5% error on 1D Burgers (vs 6.0% for gLaSDI). Produces physically consistent entropy production rates.

**LaSDI-IT:** 10^6x speedup over ALE3D simulations with <9% relative error. Greedy sampling achieves comparable accuracy with half the training data (18 vs 36 simulations).

**HLaSDI:** Achieves 1.60-7.93% displacement and 2.96-8.08% velocity errors across 1D Burgers, Wave, Telegrapher's, and Klein-Gordon equations with 10-40× speedup using 8-11 training samples.

**mLaSDI:** Reduces errors by 2.54-5.91× compared to GPLaSDI across multiscale oscillating systems, unsteady wake flow, and 1D-1V Vlasov equation, achieving 0.1-3% relative error with 40-60% less training time and fewer model parameters.

