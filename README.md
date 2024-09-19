# GPLaSDI

Numerically solving partial differential equations (PDEs) can be challenging and computationally expensive. This has led to the development of reduced-order models (ROMs) that are accurate but faster than full order models (FOMs). Recently, machine learning advances have enabled the creation of non-linear projection methods, such as Latent Space Dynamics Identification (LaSDI). LaSDI maps full-order PDE solutions to a latent space using autoencoders and learns the system of ODEs governing the latent space dynamics. By interpolating and solving the ODE system in the reduced latent space, fast and accurate ROM predictions can be made by feeding the predicted latent space dynamics into the decoder. In this paper, we introduce GPLaSDI, a novel LaSDI-based framework that relies on Gaussian process (GP) for latent space ODE interpolations. Using GPs offers two significant advantages. First, it enables the quantification of uncertainty over the ROM predictions. Second, leveraging this prediction uncertainty allows for efficient adaptive training through a greedy selection of additional training data points. This approach does not require prior knowledge of the underlying PDEs. Consequently, GPLaSDI is inherently non-intrusive and can be applied to problems without a known PDE or its residual. We demonstrate the effectiveness of our approach on the Burgers equation, Vlasov equation for plasma physics, and a rising thermal bubble problem. Our proposed method achieves between 200 and 100,000 times speed-up, with up to 7% relative error

<!-- ## Dependencies

The code requires:
* **Python 3.7.10**
* **PyTorch 1.7.1** 
* **NumPy 1.19.2** 
* **Scikit-Learn 0.24.1**
* **Scipy 1.4.1**
* **Matplotlib 3.3.4**

To run the Vlasov equation and rising bubble example, [HyPar](http://hypar.github.io/) also needs to be installed. It can be download and compiled by running:
```
git clone https://bitbucket.org/deboghosh/hypar.git
autoreconf -i
[CFLAGS="..."] [CXXFLAGS="..."] ./configure [options]
make
make install
```

## For LLNL LC Lassen users

Please install [OpenCE-1.1.2](https://lc.llnl.gov/confluence/pages/viewpage.action?pageId=678892406) -->

## Dependencies and Installation

Users can install the repository as a python package:
```
pip install .
```
This python package requires updated prerequistes:
```
"torch>=2.3.0",
"numpy>=1.26.4",
"scikit-learn>=1.4.2",
"scipy>=1.13.1",
"pyyaml>=6.0",
"matplotlib>=3.8.4",
"argparse>=1.4.0",
"h5py"
```

### For LLNL LC Lassen users

The work-in-progress python package is compatiable with [OpenCE-1.9.1](https://lc.llnl.gov/confluence/pages/viewpage.action?pageId=785286611).
For installing GPLasDI on opence environment:
```
source /your/anaconda
conda activate /your/opence/environment

conda install scikit-learn
pip install argparse

pip install .
```

## Examples

<!-- Four examples are provided, including -->

* 1D Burgers Equation

The example of Burgers 1D equation can be run:
```
cd examples
lasdi burgers1d.yml
```
Post-processing & visualization of the Burgers 1D equation can be seen in the jupyter notebook `examples/burgers1d.ipynb`.

**TODO** Support offline physics wrapper and Burgers 2D equation

* ~~2D Burgers Equation~~
* ~~1D1V Vlasov Equation~~
* ~~Rising Heat Bubble (Convection-Diffusion Equation)~~

To run the Vlasov equation and rising bubble example, [HyPar](http://hypar.github.io/) also needs to be installed. It can be download and compiled by running:
```
git clone https://bitbucket.org/deboghosh/hypar.git
autoreconf -i
[CFLAGS="..."] [CXXFLAGS="..."] ./configure [options]
make
make install
```

## Code Description

Core routines and classes are implemented in `src/lasdi` directory:

* `latent_dynamics/__init__.py`: general latent dynamics class that calibrates coefficients and predicts the latent trajectories.
  * `sindy.py`: strong SINDy class
* `physics/__init__.py`: general physics class that computes full-order model trajectories based on parameters.
  * `burgers1d.py`: Burgers1D physics equation solver (run online in python framework)
* `latent_space.py`: classes for autoencoders. Currently only vanilla multi-layer perceptron is provided.
* `param.py`: parameter space class that handles train/test parameter points.
* `gp.py`: base routines for Gaussian-process calibration and sample generation.
* `gplasdi.py`: GP-based greedy sampler class.
* `workflow.py`: controls the overall workflow of the executable `lasdi`.
* `postprocess.py`: miscellaneous post-processing and plotting routines.
* `inputs.py`: input parser class
* `fd.py`: library of high-order finite-difference stencils.
* `timing.py`: light-weight timer class

<!-- * Initial training and test data can be generated by running ```generate_data.py``` in each example directory.
* GPLaSDI models can be trained by running the file ```train_model1.py``` in each example directory.
* ```train_model1.py``` defines a **torch** autoencoder class and loads the training data and all the relevant training parameters into a ```model_parameter``` dictionnary. A ```BayesianGLaSDI``` object is created and takes into input the autoencoder and ```model_parameter```. GPLaSDI is trained by running ```BayesianGLaSDI(autoencoder, model_parameters).train()```
* ```train_framework.py``` defines the ```BayesianGLaSDI``` class, which contains the main iteration loop.
* ```utils.py``` contains FOM solvers (for the 1D/2D Burgers equations) and all the relevant functions to compute the SINDy coefficients, compute the SINDy loss, generate the GPs training datasets and train the GPs, generate ODE sample sets for new parameters, integrate the sets of ODEs, and compute the maximum variance accross the parameter space for FOM sampling. All the function in ```utils.py``` are being called in the main iteration loop in the ```BayesianGLaSDI``` class. For the 1D1V Vlasov equation and rising bubble examples, ```utils.py``` doesn't explicitly exist but the functions are instead defined in ```utils/utils_sindy.py```
* ```evaluate_model.py``` and/or ```compute_errors.py``` can be run to plot model predictions and compute relative errors between the ROM and FOM solutions

* For the 1D1V Vlasov equation and rising bubble examples, additional files are being runned within the training loop:
  * ```solver.py``` contains all the python functions to run **HyPar**. **HyPar** is a finite difference PDE solver written in C. ```init.c``` must be compiled before running GPLaSDI for the first time, using ```gcc init.c -o INIT```. ```INIT``` loads input parameter files written by ```solver.py/write_files``` and convert them into **HyPar**-readable format. Then, ```solver.py/run_hypar``` and ```solver.py/post_process_data``` run **HyPar** and convert FOM solutions into numpy arrays.
  * In the rising bubble example, an additional C file, ```PostProcess.c``` needs to be compiled before running GPLaSDI for the first time, using ```gcc PostProcess.c -o PP``` -->
 
## Citation
[Bonneville, C., Choi, Y., Ghosh, D., & Belof, J. L. (2023). GPLaSDI: Gaussian Process-based Interpretable Latent Space Dynamics Identification through Deep Autoencoder. arXiv preprint.]()

## Acknowledgement
Y. Choi was supported for this work by the CHaRMNET Mathematical Multifaceted Integrated Capability Center (MMICC).

## Licence
GPLaSDI is distributed under the terms of the MIT license. All new contributions must be made under the MIT. See
[LICENSE-MIT](https://github.com/LLNL/libROM/blob/master/LICENSE-MIT)

LLNL Release Nubmer: LLNL-CODE-850536
