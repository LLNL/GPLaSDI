from typing import List, Optional, Tuple, Union
import sys
sys.path.append('/p/gpfs1/khurana1/test/GPLaSDI')

import torch
from torch import nn
import numpy as np
import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor
from src.lasdi.latent_dynamics.sindy_high_order import SINDy
from .einsum_utils import einsum_complexhalf
from .base_spectral_conv import BaseSpectralConv
import matplotlib.pyplot as plt

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    
    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    if x.dtype == torch.complex32:
        # if x is half precision, run a specialized einsum
        return einsum_complexhalf(eq, x, weight)
    else:
        return tl.einsum(eq, x, weight)

def _contract_dense_separable(x, weight, separable):
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    return x * weight

def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, cp_weight.weights, *cp_weight.factors)
    else:
        return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        # x, y, ...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        # x, y, ...
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, tucker_weight.core, *tucker_weight.factors)
    else:
        return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, *tt_weight.factors)
    else:
        return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors. 
        if False, 
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )


Number = Union[int, float]


class SpectralConv(BaseSpectralConv):
    """SpectralConv implements the Spectral Convolution component of a Fourier layer
    described in [1]_ and [2]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : int or int tuple
        Number of modes to use for contraction in Fourier domain during training.
 
        .. warning::
            
            We take care of the redundancy in the Fourier modes, therefore, for an input 
            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N
            We will automatically keep the right amount of modes: specifically, for the 
            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes 
            as the real FFT is redundant along that last dimension. For more information on 
            mode truncation, refer to :ref:`fourier_layer_impl`

            
        .. note::

            Provided modes should be even integers. odd numbers will be rounded to the closest even number.  

        This can be updated dynamically during training.

    max_n_modes : int tuple or None, default is None
        * If not None, **maximum** number of modes to keep in Fourier Layer, along each dim
            The number of modes (`n_modes`) cannot be increased beyond that.
        * If None, all the n_modes are used.

    separable : bool, default is True
        whether to use separable implementation of contraction
        if True, contracts factors of factorized 
        tensor weight individually
    init_std : float or 'auto', default is 'auto'
        std to use for the init
    factorization : str or None, {'tucker', 'cp', 'tt'}, default is None
        If None, a single dense weight is learned for the FNO.
        Otherwise, that weight, used for the contraction in the Fourier domain
        is learned in factorized form. In that case, `factorization` is the
        tensor factorization of the parameters weight used.
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
        Ignored if ``factorization is None``
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
        Ignored if ``factorization is None``
    fft_norm : str, optional
        fft normalization parameter, by default 'forward'
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the
          factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of
          the decomposition
        Ignored if ``factorization is None``
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
        Ignored if ``factorization is None``
    complex_data: bool, optional
        whether data takes on complex values in the spatial domain, by default False
        if True, uses different logic for FFT contraction and uses full FFT instead of real-valued
    
    References
    -----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential 
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.
    
    .. [2] :

    Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
        Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024). 
        TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        physics,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
        high_order_terms = 2,
        rand_functions = [],
        test_flag = False,
        best_coefs = None,
        plot_latent = False
    ):
        super().__init__(device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.complex_data = complex_data

        # Adding sindy stuff
        self.high_order_terms = high_order_terms
        self.rand_functions = rand_functions
        self.physics = physics
        
        # Defining the sindy object. UPDATE THE SELF.LD!!!
        sindy_options = {'sindy': {'fd_type': 'sbp12', 'coef_norm_order': 2} } # finite-difference operator for computing time derivative of latent trajectory.
        self.ld = SINDy(n_modes[1]//2+1, self.high_order_terms, self.rand_functions, (n_modes[0]), sindy_options)

        # Defining terms for the coefficients and loss from latent dynamics which will be updated during the forward function!
        self.coefs = [0,0]
        self.sindy_loss = [0,0]
        self.loss_coefs = [0,0]

        # Defining a testing flag deciding when to find coeffcients for sindy or simulate using the best trained coefficients!
        self.test_flag = test_flag

        # Storing the best_coeffs if the model is in testing mode!
        self.best_coefs = best_coefs

        # Adding a flag whether to plot the latent space!
        self.plot_latent = plot_latent

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation

        self.resolution_scaling_factor = None
        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm

        if factorization is None:
            factorization = "Dense"  # No factorization

        # if separable:
        #     if in_channels != out_channels:
        #         raise ValueError(
        #             "To use separable Fourier Conv, in_channels must be equal "
        #             f"to out_channels, but got in_channels={in_channels} and "
        #             f"out_channels={out_channels}",
        #         )
        #     weight_shape = (in_channels, *max_n_modes)
        # else:
        weight_shape = (in_channels, out_channels, *max_n_modes)
        self.separable = separable

        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}

        # Create/init spectral weight tensor

        if factorization is None:
            self.weight = torch.tensor(weight_shape, dtype=torch.cfloat)
        else:
            self.weight = FactorizedTensor.new(weight_shape, rank=self.rank, 
                                     factorization=factorization, fixed_rank_modes=fixed_rank_modes,
                                     **tensor_kwargs, dtype=torch.cfloat) 
        # self.weight = torch.empty(weight_shape, dtype=torch.cfloat)
        self.weight.normal_(0, init_std)
        # nn.init.xavier_normal_(self.weight)
        self._contract = get_contract_fun(
            self.weight, implementation=implementation, separable=separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)
    
    @property
    def n_modes(self):
        return self._n_modes
    
    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int): # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # the real FFT is skew-symmetric, so the last mode has a redundacy if our data is real in space 
        # As a design choice we do the operation here to avoid users dealing with the +1
        # if we use the full FFT we cannot cut off informtion from the last mode
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(
        self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None, sindy_flag = False
    ):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient in real spatial data
        
        # FFT ON BOTH DIMENSIONS FOR TRAINING BUT ONLY SPACE IF TESTING!
        fft_dims = [-1]#list(range(-self.order, 0))


        # if self.fno_block_precision == "half":
        #     x = x.half()

        # if self.complex_data:
        #     x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
        #     dims_to_fft_shift = fft_dims
        # else: 
        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
        # When x is real in spatial domain, the last half of the last dim is redundant.
        # See :ref:`fft_shift_explanation` for discussion of the FFT shift.
        out_fft = torch.zeros(x.shape, dtype=torch.cfloat, device = x.device)
        # out_fft = x[:,:,:,:self.n_modes[1]]

        ######## The following can be adjusted to choose the first n modes or n dominant modes. Burgers1d results were derived using first n modes. Tried the following for vlasov. 
        # # Find the dominant modes!
        mag = torch.abs(x)
        average = torch.mean(mag, dim = -2, keepdim = True)
        dominant_indices = torch.argsort(average,descending=True)[0,:,:,:self.n_modes[1]].squeeze()

        out_fft[:,:,:,dominant_indices] = x[:,:,:,dominant_indices]
        # IF test flag == false!
        # if self.order > 1 and self.test_flag == False:
            # x = torch.fft.fftshift(x, dim=dims_to_fft_shift)

        # if self.fno_block_precision == "mixed":
        #     # if 'mixed', the above fft runs in full precision, but the
        #     # following operations run at half precision
        #     x = x.chalf()

        # if self.fno_block_precision in ["half", "mixed"]:
        #     out_dtype = torch.chalf
        # else:
        #     out_dtype = torch.cfloat

        # # UPDATING THE SIZE OF THE OUT_FFT HERE!
        # out_fft = torch.zeros([batchsize, self.out_channels, x.shape[2], self.n_modes[1]],
        #                       device=x.device, dtype=out_dtype)
        
        # # if current modes are less than max, start indexing modes closer to the center of the weight tensor
        # starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)]
        # # if contraction is separable, weights have shape (channels, modes_x, ...)
        # # otherwise they have shape (in_channels, out_channels, modes_x, ...)
        # if self.separable: 
        #     slices_w = [slice(None)] # channels
        # else:
        #     slices_w =  [slice(None), slice(None)] # in_channels, out_channels
        # if self.complex_data:
        #     slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts]
        # else:
        #     # The last mode already has redundant half removed in real FFT
        #     slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        #     slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        
        # weight = self.weight[slices_w]

        # ### Pick the first n_modes modes of FFT signal along each dim

        # # if separable conv, weight tensor only has one channel dim
        # if self.separable:
        #     weight_start_idx = 1
        # # otherwise drop first two dims (in_channels, out_channels)
        # else:
        #     weight_start_idx = 2
        
        # slices_x =  [slice(None), slice(None)] # Batch_size, channels

        # for all_modes, kept_modes in zip(fft_size, list(weight.shape[weight_start_idx:])):
        #     # After fft-shift, the 0th frequency is located at n // 2 in each direction
        #     # We select n_modes modes around the 0th frequency (kept at index n//2) by grabbing indices
        #     # n//2 - n_modes//2  to  n//2 + n_modes//2       if n_modes is even
        #     # n//2 - n_modes//2  to  n//2 + n_modes//2 + 1   if n_modes is odd
        #     center = all_modes // 2
        #     negative_freqs = kept_modes // 2
        #     positive_freqs = kept_modes // 2  + kept_modes % 2

        #     # this slice represents the desired indices along each dim
        #     slices_x += [slice(center - negative_freqs, center + positive_freqs)]
        
        # if weight.shape[-1] < fft_size[-1]:
        #     slices_x[-1] = slice(None, weight.shape[-1])
        # else:
        #     slices_x[-1] = slice(None)
        
        # out_fft[slices_x] = self._contract(x[slices_x], weight, separable=self.separable)

        # if self.resolution_scaling_factor is not None and output_shape is None:
        #     mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])

        # if output_shape is not None:
        #     mode_sizes = output_shape

        # # if self.order > 1:
        # #     out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])
        
        if sindy_flag:
            """
            # The following routine determines the sindy coefficients for the real and imaginary parts together.
            if self.test_flag == False and self.plot_latent == False:

                # Training Mode!
                coefs, loss_sindy, loss_coef = self.ld.calibrate(out_fft.squeeze(), self.physics.dt, compute_loss=True, numpy=False)

                # Update the terms in the class itself!
                self.coefs[0] = coefs
                self.sindy_loss[0] = loss_sindy
                self.loss_coefs[0] = loss_coef

            # Adding functionality to plot the latent space!
            elif self.test_flag == False and self.plot_latent == True:

                # Get the simulated response by extracting the initial condition!
                ic = out_fft.squeeze()[0,:].detach().numpy()
                out_fft_combined = self.ld.simulate(self.best_coefs[0], ic, self.physics.t_grid)
                plt.figure()
                plt.title('True v/s Estimated Latent Trajectory (Real)')
                plt.grid()
                plt.xlabel('Time')
                plt.ylabel('Magnitude')
                plt.plot(self.physics.t_grid, out_fft_combined[:,:self.n_modes[1]], label = 'Estimated')
                plt.plot(self.physics.t_grid, out_fft.squeeze().real.detach().numpy(), linestyle = '--',label = 'True')

                plt.figure()
                plt.grid()
                plt.xlabel('Time')
                plt.ylabel('Magnitude')
                plt.title('True v/s Estimated Latent Trajectory (Imaginary)')
                plt.plot(self.physics.t_grid, out_fft_combined[:,self.n_modes[1]:], label = 'Estimated')
                plt.plot(self.physics.t_grid, out_fft.squeeze().imag.detach().numpy(), label = 'True', linestyle = '--')

            else:

                # Testing Mode!
                out_fft_combined = self.ld.simulate(self.best_coefs[0], out_fft.squeeze().detach().numpy(), self.physics.t_grid)

                # Combine the real and imag parts!
                out_fft = out_fft_combined[:,:self.n_modes[1]] + 1j*out_fft_combined[:,self.n_modes[1]:]

                # Convert numpy back to torch!
                out_fft = torch.from_numpy(out_fft)

                # Get the size consistency!
                out_fft = (out_fft.unsqueeze(0)).unsqueeze(0)
            """
            # The following splits up the complex numbers into real and imgainry and then does the sindy fitting separately
            if self.test_flag == False and self.plot_latent == False:
                coefs, loss_sindy, loss_coef = self.ld.calibrate(out_fft[:,:,:,dominant_indices].squeeze().real, self.physics.dt, compute_loss=True, numpy=False)

                # Update the terms in the class itself!
                self.coefs[0] = coefs
                self.sindy_loss[0] = loss_sindy
                self.loss_coefs[0] = loss_coef
                #### Getting the coefficients for the imaginary part!
                coefs, loss_sindy, loss_coef = self.ld.calibrate(out_fft[:,:,:,dominant_indices].squeeze().imag, self.physics.dt, compute_loss=True, numpy=False)

                # Update the terms in the class itself!
                self.coefs[1] = coefs
                self.sindy_loss[1] = loss_sindy
                self.loss_coefs[1] = loss_coef
            elif self.test_flag == False and self.plot_latent == True:

                # Get the simulated response by extracting the initial condition!
                ic = out_fft[:,:,:,dominant_indices].squeeze()[0,:].detach().numpy()
                out_fft_real = self.ld.simulate(self.best_coefs[0], ic.real, self.physics.t_grid)
                out_fft_imag = self.ld.simulate(self.best_coefs[1], ic.imag, self.physics.t_grid)
                plt.figure()
                plt.title('True v/s Estimated Latent Trajectory (Real)')
                plt.grid()
                plt.xlabel('Time')
                plt.ylabel('Magnitude')
                plt.plot(self.physics.t_grid, out_fft_real, label = 'Estimated')
                plt.plot(self.physics.t_grid, out_fft.squeeze().real.detach().numpy(), linestyle = '--',label = 'True')
                plt.legend(['-- True','- Estimate'])
                plt.savefig('plots/real_traj.png')

                plt.figure()
                plt.grid()
                plt.xlabel('Time')
                plt.ylabel('Magnitude')
                plt.title('True v/s Estimated Latent Trajectory (Imaginary)')
                plt.plot(self.physics.t_grid, out_fft_imag, label = 'Estimated')
                plt.plot(self.physics.t_grid, out_fft.squeeze().imag.detach().numpy(), label = 'True', linestyle = '--')
                plt.legend(['-- True','- Estimate'])
                plt.savefig('plots/imag_traj.png')
            else:

                # Simulate the sindy based on the latent space and the coefficients!
                # Simulate the real part!
                out_fft_real = self.ld.simulate(self.best_coefs[0], (out_fft[:,:,:,dominant_indices].squeeze().real).detach().numpy(), self.physics.t_grid)

                # Simulate the imaginary part!
                out_fft_imag = self.ld.simulate(self.best_coefs[1], (out_fft[:,:,:,dominant_indices].squeeze().imag).detach().numpy(), self.physics.t_grid)

                # Combine the real and imaginary parts!
                out_fft = out_fft_real + 1j * out_fft_imag
                out_fft = torch.from_numpy(out_fft).to(torch.cfloat)

                # Convert the out_fft back to the same 
                out_fft_new = torch.zeros((1,1,len(self.physics.t_grid),mode_sizes[1]),device=x.device, dtype=torch.cfloat)
                out_fft_new[:,:,:,dominant_indices] == out_fft
                # Get the size consistency!
                out_fft = out_fft_new#(out_fft.unsqueeze(0)).unsqueeze(0)
        # out_fft_filtered = torch.zeros([batchsize, self.out_channels, x.shape[2], mode_sizes[1]],
        #                        device=x.device, dtype=torch.cfloat)
        
        # out_fft_filtered[...,dominant_indices] = out_fft
        
        # x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        if self.test_flag == True:
            x = torch.fft.irfftn(out_fft, s=[mode_sizes[1]], dim=fft_dims, norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(out_fft, s=[mode_sizes[1]], dim=fft_dims, norm=self.fft_norm)

        x = x.to(torch.float32)
        if self.bias is not None:
            x = x + self.bias.to(x.device)         
        return x
