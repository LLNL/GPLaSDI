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

   lasdi

