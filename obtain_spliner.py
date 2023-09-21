
import skimage.io as sio
import numpy as np
import os
import pickle
import storm_analysis.spliner.spline1D as spline1D
import storm_analysis.spliner.spline2D as spline2D
import storm_analysis.spliner.spline3D as spline3D


# load psf stack
psf_stack = sio.imread(os.path.join(os.getcwd(), 'astigmatism_08_18_with_noise.tif'))

# photon normalization
psf_stack = 1e6*psf_stack/np.sum(psf_stack, axis=(1, 2), keepdims=True)

# 1D and 2D spliners for cubic psf_stack generation
xy_splines = []
for i in range(psf_stack.shape[0]):
    xy_splines.append(spline2D.Spline2D(psf_stack[i, :, :]))  # spliners

s_size = psf_stack.shape[1]

cubic_psf_stack = np.zeros((s_size, s_size, s_size))  # target at this step
start = 0
x = start
for i in range(s_size):
    y = start
    for j in range(s_size):
        zvals = np.zeros(psf_stack.shape[0])
        for k in range(psf_stack.shape[0]):
            zvals[k] = xy_splines[k].f(y, x)

        z_spline = spline1D.Spline1D(zvals)

        max_z = float(psf_stack.shape[0]) - 1.0
        inc = max_z/(float(s_size)-1.0)
        for k in range(s_size):
            z = float(k)*inc
            if (z > max_z):
                z = max_z
            cubic_psf_stack[k, j, i] = z_spline.f(z)
        y += 1.0
    x += 1.0

# 3D Cspline analysis for PSF generation at any given axial position
spliner_3d = spline3D.Spline3D(cubic_psf_stack, verbose=True)

psf_dict = dict(pixel_size=0.11,
                zmin=0.8,
                zmax=1.8,
                type="3D",
                version=2.0,
                spline=cubic_psf_stack,
                coeff=spliner_3d.getCoeff(),
                )

with open(os.path.join(os.getcwd(), 'astigmatism_with_noise.spline'), 'wb') as fp:
    pickle.dump(psf_dict, fp)







