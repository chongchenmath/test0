"""
Solvers for matching or reconstruction with optimal information transportation.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
from odl.deform.linearized import _linear_deform
from odl.discr import Gradient
from odl.operator import IdentityOperator 

standard_library.install_aliases()


def OIT_solver_least_square(forward_op, I, data, niter, eps, lamb,
                            inverse_inertia_op, impl='mp', callback=None):
    """
    Solver for matching or shape based reconstruction using OIT with
    least square metric.
    
    Notes
    -----
    The model is:
        
        .. math:: \min_{\phi\in Diff(\Omega)} \lambda \|\sqrt{|D\phi^{-1}|} - 1\|_2^2 + \|T(\phi.I) - g)\|_2^2,
    
    where :math:`\phi.I := |D\phi^{-1}| I \circ \phi^{-1}` mass-preserving
    deformation, or :math:`\phi.I := I \circ \phi^{-1}` geometric
    deformation. If :math:`T` is identity operator, the above model reduces
    to image matching. If :math:`T` is forward projection operator,
    the above model is for image reconstruction.
    
    Parameters
    ----------
    forward_op : `Operator`
        The forward operator of imaging.
    I: `DiscreteLpElement`
        Fixed template.
    data : `DiscreteLpElement`
        The given data.
    niter : `int`
        The given maximum iteration number.
    eps : `float`
        The given step size.
    lamb : `float`
        The given regularization parameter on regularization-term side.
    inverse_inertia_op : `Operator`
        The implemantation of kernel (`poisson` or `RKHS`).
    impl : `string`, optional
        The given implementation method for group action.
        The impl chooses `mp` or `geom`, where `mp` means using
        mass-preserving method, and `geom` means geometric group action.
        Its defalt choice is `mp`.
    callback : `Class`
        Show the iterates.
    """
    # Create image space
    image_space = I.space

    # Initialize the Jacobian determinant of inverse deformation
    DPhiJacobian = image_space.one()

    # Initialize the geometric deformed template
    non_mp_deform_I = I

    # Create gradient and divergence operators
    grad_op = Gradient(image_space, method='forward', pad_mode='symmetric')
    div_op = - grad_op.adjoint

    # Create the temporary elements for update
    v = grad_op.range.element()
    
    # Create gradient of data fitting term
    gradS = forward_op.adjoint * (forward_op - data)

    # Store energy
    E = []
    kE = len(E)
    E = np.hstack((E, np.zeros(niter)))

    # Begin iteration
    for k in range(niter):

        E[k+kE] = np.asarray(lamb * (np.sqrt(DPhiJacobian) - 1) ** 2).sum()

        # Implementation for mass-preserving case
        if impl == 'mp':
            PhiStarI = DPhiJacobian * non_mp_deform_I
            grads = gradS(PhiStarI)
            tmp = grad_op(grads)
            for i in range(tmp.size):
                tmp[i] *= PhiStarI

        # Implementation for geometric case
        if impl == 'geom':
            PhiStarI = non_mp_deform_I
            grads = gradS(PhiStarI)
            tmp = - grad_op(PhiStarI)
            for i in range(tmp.size):
                tmp[i] *= grads
        
        # Compute the energy of the data fitting term         
        E[k+kE] += np.asarray((forward_op(PhiStarI) - data)**2).sum()
        
        # Show intermediate result
        if callback is not None:
            callback(PhiStarI)

        # Compute the minus L2 gradient
        u = - lamb * grad_op(np.sqrt(DPhiJacobian)) - tmp
        
        # Compute the inverse inertia
        v = inverse_inertia_op(u)

        # Check the mass
        # print(np.sum(PhiStarX))

        # Update the non-mass-preserving deformation of template
        non_mp_deform_I = image_space.element(
            _linear_deform(non_mp_deform_I, - eps * v))

        # Update the determinant of Jacobian of inverse deformation
        # Old implementation for updating Jacobian determinant
        # DPhiJacobian = np.exp(- eps * div(v)) * image_space.element(
        #    _linear_deform(DPhiJacobian, - eps * v))
        DPhiJacobian = (1.0 - eps * div_op(v)) * image_space.element(
           _linear_deform(DPhiJacobian, - eps * v))

    return PhiStarI, E


def OIT_solver_fisher_rao(I0, I1, niter, eps, lamb, inverse_inertia_op,
                          callback=None):
    """
    Solver for matching using OIT with Fisher-Rao metric.
    
    Notes
    -----
    The model is:

        .. math:: \min_{\phi\in Diff(\Omega)} \lambda \|\sqrt{|D\phi^{-1}|} - 1\|_2^2 + \|\sqrt{\phi.I_0} - \sqrt{I_1}\|_2^2,

    where :math:`\phi.I_0 := |D\phi^{-1}| I_0 \circ \phi^{-1}`
    mass-preserving deformation.

    Parameters
    ----------
    I0: `DiscreteLpElement`
        Source image.
    I1: `DiscreteLpElement`
        Target image.
    niter : `int`
        The given maximum iteration number.
    eps : `float`
        The given step size.
    lamb : `float`
        The given regularization parameter on regularization-term side.
    inverse_inertia_op : `Operator`
        The implemantation of kernel (poisson or RKHS).
    callback : `Class`
        Show the iterates.
    """
    # Get the space of I0
    domain = I0.space
    
    # Initialize the determinant of Jacobian of inverse deformation
    DPhiJacobian = domain.one()

    # Create gradient operator and divergence operator
    grad_op = Gradient(domain, method='forward', pad_mode='symmetric')
    div_op = - grad_op.adjoint
    
    # Create the temporary elements for update
    v = grad_op.range.element()

    # Initialize the non-mass-preserving deformed template
    non_mp_deform_I0 = I0
    
    inv_inertia_op = inverse_inertia_op

    # Store energy
    E = []
    kE = len(E)
    E = np.hstack((E, np.zeros(niter)))
    
#    print('Chong Chen')

    # Begin iteration
    for k in range(niter):
        # Compute the energy of the regularization term
        E[k+kE] = np.asarray(lamb * (np.sqrt(DPhiJacobian) - 1) ** 2).sum()

        # Implementation for mass-preserving case
        PhiStarI0 = DPhiJacobian * non_mp_deform_I0

        # Show intermediate result
        if callback is not None:
            callback(PhiStarI0)

        # For Fisher-Rao distance
        sqrt_mp_I0 = np.sqrt(PhiStarI0)
        sqrt_I1 = np.sqrt(I1)
        grad_sqrt_mp_I0 = grad_op(sqrt_mp_I0)
        grad_sqrt_I1 = grad_op(sqrt_I1)
        
        # Compute the energy of the data fitting term         
        E[k+kE] += np.asarray((sqrt_mp_I0 - sqrt_I1)**2).sum()

        # Compute the L2 gradient of the data fitting term
        grad_fitting = grad_op.range.zero()
        for i in range(grad_op.range.size):
            grad_fitting[i] = sqrt_I1 * grad_sqrt_mp_I0[i] - \
                sqrt_mp_I0 * grad_sqrt_I1[i]
                
        # Compute the minus L2 gradient
        u = - lamb * grad_op(np.sqrt(DPhiJacobian)) - grad_fitting

        # Compute inverse inertia
        v = inv_inertia_op(u)

        # Update the non-mass-preserving deformed template
        non_mp_deform_I0 = domain.element(
            _linear_deform(non_mp_deform_I0, - eps * v))

        # Implementation for updating Jacobian determinant
        DPhiJacobian = (1.0 - eps * div_op(v)) * domain.element(
           _linear_deform(DPhiJacobian, - eps * v))
    
    return PhiStarI0, E


def OIT_solver(forward_op, template, data, niter, eps, lamb,
               inverse_inertia_op, aim='matching', metric='fisher_rao',
               group_action='mp', callback=None):
    """
    Solver for image matching or shape based image reconstruction using OIT.
    
    The regularization term in model is:
        
        .. math:: \lambda \|\sqrt{|D\phi^{-1}|} - 1\|^2
    
    where :math:`|D\phi^{-1}|` is the Jacobian determinant of the inverse
    of diffeomorphic deformation.
    
    Parameters
    ----------
    forward_op : `Operator`
        The forward operator of imaging.
    template : `DiscreteLpElement`
        Fixed template.
    data : `DiscreteLpElement`
        The given data.
    niter : `int`
        The given maximum iteration number.
    eps : `float`
        The given step size.
    lamb : `float`
        The given regularization parameter on regularization-term side.
    inverse_inertia_op : `Operator`
        The implemantation of kernel (poisson or RKHS).
    aim : `string`, optional
        The given implementation method for image matching or reconstruction.
        The defalt is image matching.
    metric : `string`, optional
        The given implementation method for the fitting term in model.
        The defalt is Fisher-Rao metric.
    group_action : `string`, optional
        The given implementation method for group action.
        The defalt is mass-preserving group action.
    callback : `Class`
        Show the iterates.
    """
    if aim == 'matching':
        if not isinstance(forward_op, IdentityOperator):
            raise ValueError('The `forward_op` should be `IdentityOperator` '
                            'in image space for matching')
        if not template.space == data.space:
            raise ValueError('The `template.space` should be the same as '
                             '`data.space` for matching')
        if metric == 'least_square':
            rec_result, E = OIT_solver_least_square(
                    forward_op, template, data, niter, eps, lamb,
                    inverse_inertia_op, group_action, callback)
        elif metric == 'fisher_rao':
            if not group_action == 'mp':
                raise ValueError('The `Fisher-Rao` metric needs to be used '
                                 'with `mass-preserving` group action '
                                 'for matching')
            rec_result, E = OIT_solver_fisher_rao(
                    template, data, niter, eps, lamb, inverse_inertia_op,
                    callback)
        else:
            raise NotImplementedError('now only support `least square` or '
                                      '`fisher_rao` metric')
    elif aim == 'reconstruction':
        if isinstance(forward_op, IdentityOperator):
            raise ValueError('The `forward_op` should be `Radon transform` '
                            'in image space for reconstruction')
        if template.space == data.space:
            raise ValueError('The `template.space` should not be the same as '
                             '`data.space` for reconstruction')
        if metric == 'least_square':
            rec_result, E = OIT_solver_least_square(
                    forward_op, template, data, niter, eps, lamb,
                    inverse_inertia_op, group_action, callback)
        else:
            raise NotImplementedError('now only support `least square` metric')
    else:
        raise NotImplementedError('now only support the aim of `matching` or '
                                      'reconstruction')
    
    return rec_result, E
