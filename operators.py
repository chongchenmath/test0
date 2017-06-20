"""
Operators and functions for reconstruction using OIT.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
from odl.trafos import FourierTransform
from odl.operator import DiagonalOperator
from odl.discr import ResizingOperator

standard_library.install_aliases()


def snr(signal, noise, impl):
    """Compute the signal-to-noise ratio.

    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = variance(signal) / variance(noise),
        'dB' means SNR = 10 * log10 (variance(signal) / variance(noise)).

    Returns
    -------
    snr : `float`
        Value of signal-to-noise ratio.
        If the power of noise is zero, then the return is 'inf',
        otherwise, the computed value.
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal) / signal.size
        ave2 = np.sum(noise) / noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')


def show_result(template, ground_truth, rec_result, nonoise_data, noise_data,
                aim, energy=None):
    """Show results.

    Parameters
    ----------
    template : `DiscreteLpElement`
        Fixed template.
    ground_truth : `DiscreteLpElement`
        Given target
    rec_result : `DiscreteLpElement`
        Result.
    data : `DiscreteLpElement`
        Noise-free data.
    noise_data : `DiscreteLpElement`
        Used data.
    aim : 'string', optional
        The given implementation method for image matching or reconstruction.
    energy : `array-like`
        Energy for each iteration.
    """
    
    if aim == 'reconstruction':
        plt.figure(1, figsize=(16, 16))
        plt.clf()
        
        plt.subplot(2, 2, 1)
        plt.imshow(np.rot90(template), cmap='bone',
                   vmin=np.asarray(template).min(),
                   vmax=np.asarray(template).max())
        plt.axis('off')
        plt.title('template')
        
        plt.subplot(2, 2, 2)
        plt.imshow(np.rot90(rec_result), cmap='bone',
                   vmin=np.asarray(rec_result).min(),
                   vmax=np.asarray(rec_result).max())
        plt.axis('off')
        plt.title('rec_result')
        
        plt.subplot(2, 2, 3)
        plt.imshow(np.rot90(ground_truth), cmap='bone',
                   vmin=np.asarray(ground_truth).min(),
                   vmax=np.asarray(ground_truth).max())
        plt.axis('off')
        plt.title('ground truth')
        
        plt.subplot(2, 2, 4)
        plt.plot(np.asarray(nonoise_data)[0], 'b', linewidth=1.0)
        plt.plot(np.asarray(noise_data)[0], 'r', linewidth=0.5)
        plt.axis([0, nonoise_data.shape[1] - 1, -1, 9])
        plt.grid(True, linestyle='--')
        
        if energy is not None:
            plt.figure(2, figsize=(8, 1.5))
            plt.clf()
            plt.plot(energy)
            plt.ylabel('Energy')
            # plt.gca().axes.yaxis.set_ticklabels(['0']+['']*8)
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.grid(True)

    elif aim == 'matching':
        plt.figure(1, figsize=(20, 10))
        plt.clf()
        
        plt.subplot(1, 3, 1)
        plt.imshow(np.rot90(template), cmap='bone',
                   vmin=np.asarray(template).min(),
                   vmax=np.asarray(template).max())
        plt.axis('off')
        plt.title('template')
        
        plt.subplot(1, 3, 2)
        plt.imshow(np.rot90(rec_result), cmap='bone',
                   vmin=np.asarray(rec_result).min(),
                   vmax=np.asarray(rec_result).max())
        plt.axis('off')
        plt.title('match_result')
        
        plt.subplot(1, 3, 3)
        plt.imshow(np.rot90(ground_truth), cmap='bone',
                   vmin=np.asarray(ground_truth).min(),
                   vmax=np.asarray(ground_truth).max())
        plt.axis('off')
        plt.title('ground truth')
        
        if energy is not None:
            plt.figure(2, figsize=(8, 1.5))
            plt.clf()
            plt.plot(energy)
            plt.ylabel('Energy')
            # plt.gca().axes.yaxis.set_ticklabels(['0']+['']*8)
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.grid(True)

    else:
        raise NotImplementedError('now only support `reconstruction` or '
                                      '`matching`')


def padded_ft_op(space, padded_size):
    """Create zero-padding FT.

    Parameters
    ----------
    space : `DiscreteLp`
        The space needs to do FT.
    padded_size : positive `int`
        The padded size for zero padding.
    
    Returns
    -------
    padded_ft_op : `operator`
        Operator of FT composing with padded operator.
    """
    padding_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = FourierTransform(
        padding_op.range, halfcomplex=False, shift=shifts)

    return ft_op * padding_op


def vectorized_padded_ft_op(space, padded_size):
    """Create vectorial zero-padding FT.

    Parameters
    ----------
    space : `DiscreteLp`
        The space needs to do FT.
    padded_size : positive `int`
        The padded size for zero padding.
    
    Returns
    -------
    vectorized_padded_ft_op : `operator`
        The vectorized padded FT operator with the space dimension.
    """
    pad_ft_op = padded_ft_op(space, padded_size)
    return DiagonalOperator(*([pad_ft_op] * space.ndim))


def vectorized_kernel(space, kernel_func):
    """Compute the vectorized discrete kernel ``K``.
    
    Parameters
    ----------
    space : `DiscreteLp`
        The space needs to do FT.
    kernel_func : `function`
        The used kernel function.
    
    Returns
    -------
    vectorized_discretized_kernel : `ProductSpaceElement`
        The vectorized discrete kernel with the space dimension.
    """
    kspace = space.tangent_bundle
    # Create the array of kernel values on the grid points
    vectorized_discretized_kernel = kspace.element(
        [space.element(kernel_func) for _ in range(space.ndim)])
    return vectorized_discretized_kernel


def kernel_ft(space, padded_size, kernel_func):
    """Compute the zero-padding FT of the kernel function.

    Parameters
    ----------
    space : `DiscreteLp`
        The space needs to do FT.
    padded_size : positive `int`
        The padded size for zero padding.
    kernel_func : `function`
        The used kernel function.
        
    Returns
    -------
    kernel_ft : `DiscreteLpElement`
        The zero-padding FT of the kernel function.
    """
    # Create the array of kernel values on the space-grid points
    discretized_kernel = space.element(kernel_func)
    pad_ft_op = padded_ft_op(space, padded_size)
    return pad_ft_op(discretized_kernel)


def vectorized_kernel_ft(space, padded_size, kernel_func):
    """Compute the vectorial zero-padding FT of the kernel function.

    Parameters
    ----------
    space : `DiscreteLp`
        The space needs to do FT.
    padded_size : positive `int`
        The padded size for zero padding.
    kernel_func : `function`
        The used kernel function.
    
    Returns
    -------
    vectorized_kernel_ft : `ProductSpaceElement`
        The vectorized zero-padding FT of the kernel function.
    """
    vectorized_discretized_kernel = vectorized_kernel(space, kernel_func)
    vec_ft_op = vectorized_padded_ft_op(space, padded_size)
    return vec_ft_op(vectorized_discretized_kernel)


def poisson_kernel_ft(space, padded_size):
    """Compute the zero-padding FT of the inverse Laplacian.

    Parameters
    ----------
    space : `DiscreteLp`
        The space needs to do FT.
    padded_size : positive `int`
        The padded size for zero padding.

    Returns
    -------
    poisson_kernel_ft : `DiscreteLpElement`
        The zero-padding FT of the inverse Laplacian.
    """
    pad_ft_op = padded_ft_op(space, padded_size)
    k2 = np.sum((pad_ft_op.range.points() ** 2).T, axis=0)
    k2 = pad_ft_op.range.element(np.maximum(np.abs(k2), 0.0000001))
    inv_k2 = 1 / k2
    return pad_ft_op.range.element(np.minimum(np.abs(inv_k2), 200))


def vectorized_poisson_kernel_ft(space, padded_size):
    """Compute the vectorial zero-padding FT of the inverse Laplacian.

    Parameters
    ----------
    space : `DiscreteLp`
        The space needs to do FT.
    padded_size : positive `int`
        The padded size for zero padding.

    Returns
    -------
    vectorized_poisson_kernel_ft : `ProductSpaceElement`
        The vectorized zero-padding FT of the inverse Laplacian.
    """
    pad_ft_op = padded_ft_op(space, padded_size)
    k2 = np.sum((pad_ft_op.range.points() ** 2).T, axis=0)
    k2 = pad_ft_op.range.element(np.maximum(np.abs(k2), 0.0000001))
    inv_k2 = 1 / k2
    inv_k2 = pad_ft_op.range.element(np.minimum(np.abs(inv_k2), 200))

    kspace = pad_ft_op.range.tangent_bundle
    return kspace.element([inv_k2 for _ in range(space.ndim)])


def inverse_inertia_op(impl, space, padded_size, kernel_func=None):
    """Create the inverse inertia operator.

    Parameters
    ----------
    impl3 : `string`
        implementation method, solving poisson equation or using RKHS.
    space : `DiscreteLp`
        The space needs to do FT.
    padded_size : positive `int`
        The padded size for zero padding.
    kernel_func : `function`, optional
        The used kernel function.
        
    Returns
    -------
    inverse_inertia_op : `operator`
        The inverse of inertia operator
    """ 
    temp = (2 * np.pi) ** (space.ndim / 2.0)
    vec_ft_op = vectorized_padded_ft_op(space, padded_size)

    if impl == 'poisson':
        kernel_ft = vectorized_poisson_kernel_ft(space, padded_size)
        return temp * vec_ft_op.inverse * kernel_ft * vec_ft_op
    
    elif impl == 'rkhs':
        if kernel_func == None:
            raise NotImplementedError('unknown kernel function')
        kernel_ft  = vectorized_kernel_ft(space, padded_size, kernel_func)
        return temp * vec_ft_op.inverse * kernel_ft * vec_ft_op    
