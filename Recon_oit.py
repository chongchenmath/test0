from __future__ import print_function, division, absolute_import
from future import standard_library
import odl
import matplotlib.pyplot as plt
import numpy as np
from optimal_information_transport import OIT_solver
from operators import snr, show_result, inverse_inertia_op

standard_library.install_aliases()


# Define Gaussian kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))


# Give input images
I0name = './pictures/j.png' # 64 * 64 ---> 92
I1name = './pictures/v.png' # 64 * 64 ---> 92
#I0name = './pictures/handnew2.png' # 256 * 256 ---> 362
#I1name = './pictures/handnew1.png' # 256 * 256 ---> 362

## Get digital images
I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)

# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(min_pt=[-16, -16], max_pt=[16, 16], shape=[64, 64],
    dtype='float32', interp='linear')

# Create the ground truth as the given image
ground_truth = space.element(I1)

# Create the ground truth as the Shepp-Logan phantom
# ground_truth = shepp_logan(space, modified=True)

# # Create the ground truth as the submarine phantom
# ground_truth = odl.util.submarine_phantom(space, smooth=True, taper=50.0)

# Create the template as the given image
template = space.element(I0)

# # Create the template as the disc phantom
# template = odl.util.disc_phantom(space, smooth=True, taper=50.0)

## Create the template for Shepp-Logan phantom
#deform_field_space = space.tangent_bundle
#disp_func = [
#    lambda x: 16.0 * np.sin(np.pi * x[0] / 40.0),
#    lambda x: 16.0 * np.sin(np.pi * x[1] / 36.0)]
#deform_field = deform_field_space.element(disp_func)
#template = space.element(geometric_deform(
#    shepp_logan(space, modified=True), deform_field))

# Maximum iteration number
niter = 400

# The group_action chooses 'mp' or 'geom'
# 'mp' means mass-preserving group action,
# 'geom' means geometric group action.
group_action = 'mp'

# The aim chooses 'matching' or 'reconstruction'
# 'matching' means image matching,
# 'reconstruction' means image reconstruction
aim = 'matching'

# The impl3 chooses 'poisson' or 'rkhs', 'poisson' means using poisson solver,
# 'rkhs' means using V-gradient
ker_style = 'poisson'

# The metric chooses 'least_square' or 'fisher_rao', 'least_square' means using
# l2-norm least square fitting term, 'fisher_rao'  meams using Fisher-Rao
# fitting term 
metric = 'fisher_rao'

# Show intermiddle results
callback = odl.solvers.CallbackShow(
    '{!r} {!r} iterates'.format(aim, group_action), display_step=5) & \
    odl.solvers.CallbackPrintIteration()

# Normalize the template's density as the same as the ground truth if consider
# mass preserving method
if group_action == 'mp':
#    template *= np.sum(ground_truth) / np.sum(template)
    template *= np.linalg.norm(ground_truth, 'fro')/ \
        np.linalg.norm(template, 'fro')

# Show ground truth and template
ground_truth.show('Ground truth')
template.show('Template')

# Give step size for solver
eps = 0.02

# Give regularization parameter
lamb = 0.05

# Fix the sigma parameter for Gaussian kernel
sigma = 5.0

if aim == 'reconstruction':
    # Give the number of directions
    num_angles = 10
    
    # Create the uniformly distributed directions
    angle_partition = odl.uniform_partition(0, np.pi, num_angles,
                                            nodes_on_bdry=[(True, False)])
    
    # Create 2-D projection domain
    # The length should be 1.5 times of that of the reconstruction space
    detector_partition = odl.uniform_partition(-24, 24, 92)
    
    # Create 2-D parallel projection geometry
    geometry = odl.tomo.Parallel2dGeometry(angle_partition,
                                           detector_partition)
    
    # Ray transform aka forward projection. We use ASTRA CUDA backend.
    op = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

elif aim == 'matching':
    # Create the forward operator for image matching
    op = odl.IdentityOperator(space)
    

# Create projection data by calling the ray transform on the phantom
data = op(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise = odl.phantom.white_noise(op.range) * 0.0

# Create the noisy projection data
noise_data = data + noise

# Compute the signal-to-noise ratio in dB
snr = snr(data, noise, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Give the padded size for FT
padded_size = 2 * space.shape[0]

# Implement different gradient (poisson or RKHS)
inv_intia_op = inverse_inertia_op(ker_style, space, padded_size,
                                  kernel_func=kernel)

# Compute by optimal information transport solver
rec_result, E = OIT_solver(op, template, noise_data, niter, eps, lamb,
                           inv_intia_op, aim, metric, group_action, callback)

# Show result
show_result(template, ground_truth, rec_result, data,
            noise_data, aim, energy=E)
