# ------------------------------------------------------------ #
#
# file : preprocessing/normalisation.py
# author : CM
#
# ------------------------------------------------------------ #
import numpy as np

# Rescaling (min-max normalization)
def linear_intensity_normalization(loaded_dataset):
    loaded_dataset = (loaded_dataset / loaded_dataset.max())
    return loaded_dataset

# Preprocess dataset with intensity normalisation
# (zero mean and unit variance)
def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std  = dataset.std()
    return ((dataset - mean) / std).astype(dtype)

# Intensities normalized to the range [0, 1]
def intensityNormalisationFeatureScaling(dataset, dtype):
    max = dataset.max()
    min = dataset.min()

    return ((dataset - min) / (max - min)).astype(dtype)

# Intensity max clipping with c "max value"
def intensityMaxClipping(dataset, c, dtype):
    return np.clip(a=dataset, a_min=0, a_max=c).astype(dtype)

# Intensity projection
def intensityProjection(dataset, p, dtype):
    return (dataset ** p).astype(dtype)