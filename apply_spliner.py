
import os
import storm_analysis.spliner.spline_to_psf as splineToPSF
import numpy as np
import skimage
import matplotlib.pyplot as plt

# load spline file and construct the spliner
cwd = os.getcwd()
spline_file = os.path.join(cwd, 'astigmatism_with_noise.spline')
spliner = splineToPSF.SplineToPSF3D(spline_file)


# ground truth for comparison
psf_stack = skimage.io.imread(os.path.join(os.getcwd(), 'astigmatism_08_18_with_noise.tif'))
psf_stack = 1e6*psf_stack/np.sum(psf_stack, axis=(1, 2), keepdims=True)


zs = np.linspace(0.8, 1.8, psf_stack.shape[0])
all_psfs = np.zeros(psf_stack.shape)

for i in range(zs.shape[0]):
    all_psfs[i, :, :] = spliner.getPSF(zs[i], shape=psf_stack.shape[1:])  # psf generation

all_psfs = 1e6*all_psfs/np.sum(all_psfs, axis=(1, 2), keepdims=True)


idx = 20  # which image to show

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
ax.set_title('(a) Measured PSF')
plt.imshow(psf_stack[idx, :, :])
plt.colorbar()
plt.axis('off')

ax = fig.add_subplot(2, 2, 3)
plt.imshow(all_psfs[idx, :, :])
ax.set_title('(c) PSF from spliner')
plt.colorbar()
plt.axis('off')


difference = np.abs(psf_stack[idx, :, :]-all_psfs[idx, :, :])
ax = fig.add_subplot(2, 2, 4)
plt.imshow(difference)
plt.colorbar()
ax.set_title('(d) Difference')
plt.axis('off')

plt.show()




