# ------------------------------------------------------------ #
#
# file : utils/io/write.py
# author : CM
# Function to write results
#
# ------------------------------------------------------------ #

import nibabel as nib
import numpy as np

# write nii file from a numpy 3d array
def npToNii(data, filename):
    axes = np.eye(4)
    axes[0][0] = -1
    axes[1][1] = -1
    image = nib.Nifti1Image(data, axes)
    nib.save(image, filename)

# write nii file from a numpy 3d array with affine configuration
def npToNiiAffine(data, affine, filename):
    image = nib.Nifti1Image(data, affine)
    nib.save(image, filename)