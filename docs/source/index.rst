.. LaSDI documentation master file, created by
   sphinx-quickstart on Wed Oct 16 22:11:53 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LaSDI documentation
===================

LaSDI is a light-weight python package for Latent Space Dynamics Identification.
LaSDI maps full-order PDE solutions to a latent space using autoencoders and learns the system of ODEs governing the latent space dynamics.
By interpolating and solving the ODE system in the reduced latent space, fast and accurate ROM predictions can be made by feeding the predicted latent space dynamics into the decoder.
It also supports parametric interpolation of latent dynamics according to uncertainties evaluated via Gaussian process.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

References
===================

* Bonneville, Christophe, Xiaolong He, April Tran, Jun Sur Park, William Fries, Daniel A. Messenger, Siu Wun Cheung et al. "A Comprehensive Review of Latent Space Dynamics Identification Algorithms for Intrusive and Non-Intrusive Reduced-Order-Modeling." arXiv preprint arXiv:2403.10748 (2024).
* Fries, William D., Xiaolong He, and Youngsoo Choi. "LaSDI: Parametric latent space dynamics identification." Computer Methods in Applied Mechanics and Engineering 399 (2022): 115436.
* He, Xiaolong, Youngsoo Choi, William D. Fries, Jonathan L. Belof, and Jiun-Shyan Chen. "gLaSDI: Parametric physics-informed greedy latent space dynamics identification." Journal of Computational Physics 489 (2023): 112267.
* Tran, April, Xiaolong He, Daniel A. Messenger, Youngsoo Choi, and David M. Bortz. "Weak-form latent space dynamics identification." Computer Methods in Applied Mechanics and Engineering 427 (2024): 116998.
* Park, Jun Sur Richard, Siu Wun Cheung, Youngsoo Choi, and Yeonjong Shin. "tLaSDI: Thermodynamics-informed latent space dynamics identification." arXiv preprint arXiv:2403.05848 (2024).
* Bonneville, Christophe, Youngsoo Choi, Debojyoti Ghosh, and Jonathan L. Belof. "Gplasdi: Gaussian process-based interpretable latent space dynamics identification through deep autoencoder." Computer Methods in Applied Mechanics and Engineering 418 (2024): 116535.
* He, Xiaolong, April Tran, David M. Bortz, and Youngsoo Choi. "Physics-informed active learning with simultaneous weak-form latent space dynamics identification." arXiv preprint arXiv:2407.00337 (2024).
