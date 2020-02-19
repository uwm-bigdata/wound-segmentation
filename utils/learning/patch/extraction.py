# ------------------------------------------------------------ #
#
# file : utils/learning/patch/extraction.py
# author : CM
# Function to extract patch from input dataset
#
# ------------------------------------------------------------ #
import sys
from random import randint
import numpy as np
from keras.utils import to_categorical

# ----- Patch Extraction -----
# -- Single Patch
# exctract a patch from an image
def extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z):
    patch = d[x:x+patch_size_x,y:y+patch_size_y,z:z+patch_size_z]
    return patch

# extract a patch from an image. The patch can be out of the image (0 padding)
def extractPatchOut(d, patch_size_x, patch_size_y, patch_size_z, x_, y_, z_):
    patch = np.zeros((patch_size_x, patch_size_y, patch_size_z), dtype='float16')
    for x in range(0,patch_size_x):
        for y in range(0, patch_size_y):
            for z in range(0, patch_size_z):
                if(x+x_ >= 0 and x+x_ < d.shape[0] and y+y_ >= 0 and y+y_ < d.shape[1] and z+z_ >= 0 and z+z_ < d.shape[2]):
                    patch[x,y,z] = d[x+x_,y+y_,z+z_]
    return patch

# create random patch for an image
def generateRandomPatch(d, patch_size_x, patch_size_y, patch_size_z):
    x = randint(0, d.shape[0]-patch_size_x)
    y = randint(0, d.shape[1]-patch_size_y)
    z = randint(0, d.shape[2]-patch_size_z)
    data = extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z)
    return data

# -- Multiple Patchs
# create random patchs for an image
def generateRandomPatchs(d, patch_size_x, patch_size_y, patch_size_z, patch_number):
    # max_patch_nb = (d.shape[0]-patch_size_x)*(d.shape[1]-patch_size_y)*(d.shape[2]-patch_size_z)
    data = np.empty((patch_number, patch_size_x, patch_size_y, patch_size_z), dtype='float16')

    for i in range(0,patch_number):
        data[i] = generateRandomPatch(d, patch_size_x, patch_size_y, patch_size_z)

    return data

# divide the full image into patchs
# todo : missing data if shape%patch_size is not 0
def generateFullPatchs(d, patch_size_x, patch_size_y, patch_size_z):
    patch_nb = int((d.shape[0]/patch_size_x)*(d.shape[1]/patch_size_y)*(d.shape[2]/patch_size_z))
    data = np.empty((patch_nb, patch_size_x, patch_size_y, patch_size_z), dtype='float16')
    i = 0
    for x in range(0,d.shape[0], patch_size_x):
        for y in range(0, d.shape[1], patch_size_y):
            for z in range(0,d.shape[2], patch_size_z):
                data[i] = extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z)
                i = i+1

    return data

def generateFullPatchsPlus(d, patch_size_x, patch_size_y, patch_size_z, dx, dy, dz):
    patch_nb = int((d.shape[0]/dx)*(d.shape[1]/dy)*(d.shape[2]/dz))
    data = np.empty((patch_nb, patch_size_x, patch_size_y, patch_size_z), dtype='float16')
    i = 0
    for x in range(0,d.shape[0]-dx, dx):
        for y in range(0, d.shape[1]-dy, dy):
            for z in range(0,d.shape[2]-dz, dz):
                data[i] = extractPatch(d, patch_size_x, patch_size_y, patch_size_z, x, y, z)
                i = i+1

    return data

def noNeg(x):
    if(x>0):
        return x
    else:
        return 0

def generateFullPatchsCentered(d, patch_size_x, patch_size_y, patch_size_z):
    patch_nb = int(2*(d.shape[0]/patch_size_x)*2*(d.shape[1]/patch_size_y)*2*(d.shape[2]/patch_size_z))
    data = np.zeros((patch_nb, patch_size_x, patch_size_y, patch_size_z), dtype='float16')
    i = 0
    psx = int(patch_size_x/2)
    psy = int(patch_size_y/2)
    psz = int(patch_size_z/2)
    for x in range(-int(patch_size_x/4),d.shape[0]-3*int(patch_size_x/4)+1, psx):
        for y in range(-int(patch_size_y/4), d.shape[1]-3*int(patch_size_y/4)+1, psy):
            for z in range(-int(patch_size_z/4),d.shape[2]-3*int(patch_size_z/4)+1, psz):
                # patch = np.zeros((psx,psy,psz), dtype='float16')
                # patch = d[noNeg(x):noNeg(x)+patch_size_x,noNeg(y):noNeg(y)+patch_size_y,noNeg(z):noNeg(z)+patch_size_z]
                patch = extractPatchOut(d,patch_size_x, patch_size_y, patch_size_z, x, y, z)
                data[i] = patch
                i = i+1
    return data

# ----- Patch Extraction Generator -----
# Generator of random patchs of size 32*32*32
def generatorRandomPatchs(features, labels, batch_size, patch_size_x, patch_size_y, patch_size_z):
    batch_features = np.zeros((batch_size, patch_size_x, patch_size_y, patch_size_z, features.shape[4]), dtype='float16')
    batch_labels = np.zeros((batch_size, patch_size_x, patch_size_y, patch_size_z, labels.shape[4]), dtype='float16')

    while True:
        for i in range(batch_size):
            id = randint(0,features.shape[0]-1)
            x = randint(0, features.shape[1]-patch_size_x)
            y = randint(0, features.shape[2]-patch_size_y)
            z = randint(0, features.shape[3]-patch_size_z)

            batch_features[i]   = extractPatch(features[id], patch_size_x, patch_size_y, patch_size_z, x, y, z)
            batch_labels[i]     = extractPatch(labels[id], patch_size_x, patch_size_y, patch_size_z, x, y, z)

        yield batch_features, batch_labels

# Generator of random patchs of size 32*32*32 and 16*16*16
def generatorRandomPatchs3216(features, labels, batch_size):
    batch_features = np.zeros((batch_size, 32, 32, 32, features.shape[4]), dtype='float16')
    batch_labels = np.zeros((batch_size, 16, 16, 16, labels.shape[4]), dtype='float16')

    while True:
        for i in range(batch_size):
            id = randint(0,features.shape[0]-1)
            x = randint(0, features.shape[1]-32)
            y = randint(0, features.shape[2]-32)
            z = randint(0, features.shape[3]-32)

            batch_features[i]   = extractPatch(features[id], 32, 32, 32, x, y, z)
            batch_labels[i]     = extractPatch(labels[id], 16, 16, 16, x+16, y+16, z+16)

        yield batch_features, batch_labels

def generatorRandomPatchsLabelCentered(features, labels, batch_size, patch_size_x, patch_size_y, patch_size_z):
    patch_centered_size_x = int(patch_size_x/2)
    patch_centered_size_y = int(patch_size_y/2)
    patch_centered_size_z = int(patch_size_z/2)

    batch_features = np.zeros((batch_size, patch_size_x, patch_size_y, patch_size_z, features.shape[4]), dtype=features.dtype)
    batch_labels = np.zeros((batch_size, patch_centered_size_x, patch_centered_size_y, patch_centered_size_z,
                             labels.shape[4]), dtype=labels.dtype)

    while True:
        for i in range(batch_size):
            id = randint(0,features.shape[0]-1)
            x = randint(0, features.shape[1]-patch_size_x)
            y = randint(0, features.shape[2]-patch_size_y)
            z = randint(0, features.shape[3]-patch_size_z)

            batch_features[i]   = extractPatch(features[id], patch_size_x, patch_size_y, patch_size_z, x, y, z)
            batch_labels[i]     = extractPatch(labels[id], patch_centered_size_x, patch_centered_size_y, patch_centered_size_z,
                                               int(x+patch_size_x/4), int(y+patch_size_y/4), int(z+patch_size_z/4))

        yield batch_features, batch_labels

def generatorRandomPatchsDolz(features, labels, batch_size, patch_size_x, patch_size_y, patch_size_z):
    batch_features = np.zeros((batch_size, patch_size_x, patch_size_y, patch_size_z, features.shape[4]), dtype=features.dtype)
    batch_labels   = np.zeros((batch_size, int(patch_size_x / 2) * int(patch_size_y / 2) * int(patch_size_z / 2), 2), dtype=labels.dtype)

    while True:
        for i in range(batch_size):
            id = randint(0,features.shape[0]-1)
            x = randint(0, features.shape[1]-patch_size_x)
            y = randint(0, features.shape[2]-patch_size_y)
            z = randint(0, features.shape[3]-patch_size_z)

            batch_features[i] = extractPatch(features[id], patch_size_x, patch_size_y, patch_size_z, x, y, z)
            tmpPatch = extractPatch(labels[id], int(patch_size_x/2), int(patch_size_y/2), int(patch_size_z/2),
                                    int(x+patch_size_x/4), int(y+patch_size_y/4), int(z+patch_size_z/4))
            batch_labels[i] = to_categorical(tmpPatch.flatten(),2)
            """
            count = 0
            for x in range(0, tmpPatch.shape[0]):
                for y in range(0, tmpPatch.shape[1]):
                    for z in range(0, tmpPatch.shape[2]):
                        if(tmpPatch[x,y,z,0] == 1):
                            batch_labels[i,count,1] = 1
                        else:
                            batch_labels[i,count,0] = 1
                        count += 1
            """
        yield batch_features, batch_labels

from scipy.ndimage import zoom, rotate
# Generate random patchs with random linear transformation
# translation (random position) rotation, scale
# Preconditions :   patch_features_ % patch_labels_ = 0
#                   patch_features_ >= patch_labels_
# todo : scale
def generatorRandomPatchsLinear(features, labels, patch_features_x, patch_features_y, patch_features_z,
                                patch_labels_x, patch_labels_y, patch_labels_z):

    patch_features = np.zeros((1, patch_features_x, patch_features_y, patch_features_z, features.shape[4]), dtype=features.dtype)
    patch_labels   = np.zeros((1, patch_labels_x, patch_labels_y, patch_labels_z, labels.shape[4]), dtype=labels.dtype)

    if(patch_features_x % patch_labels_x != 0 or patch_features_y % patch_labels_y != 0 or patch_features_z % patch_labels_z != 0):
        sys.exit(0x00F0)

    if(patch_features_x < patch_labels_x or patch_features_y < patch_labels_y or patch_features_z < patch_labels_z):
        sys.exit(0x00F1)

    # middle of patch
    mx = int(patch_features_x/2)
    my = int(patch_features_y/2)
    mz = int(patch_features_z/2)
    # patch label size/2
    sx = int(patch_labels_x / 2)
    sy = int(patch_labels_y / 2)
    sz = int(patch_labels_z / 2)

    while True:
        id = randint(0, features.shape[0]-1)
        x  = randint(0, features.shape[1]-patch_features_x)
        y  = randint(0, features.shape[2]-patch_features_y)
        z  = randint(0, features.shape[3]-patch_features_z)

        # todo : check time consumtion and rotation directly on complete image
        r0 = randint(0, 360)-180
        r1 = randint(0, 360)-180
        r2 = randint(0, 360)-180
        rot_features = rotate(input=features[0], angle=r0, axes=(0, 1), reshape=False)
        rot_features = rotate(input=rot_features, angle=r1, axes=(1, 2), reshape=False)
        rot_features = rotate(input=rot_features, angle=r2, axes=(2, 0), reshape=False)
        rot_labels   = rotate(input=labels[0], angle=r0, axes=(0, 1), reshape=False)
        rot_labels   = rotate(input=rot_labels, angle=r1, axes=(1, 2), reshape=False)
        rot_labels   = rotate(input=rot_labels, angle=r2, axes=(2, 0), reshape=False)

        patch_features[0] = extractPatch(rot_features, patch_features_x, patch_features_y, patch_features_z, x, y, z)

        patch_labels[0]   = extractPatch(rot_labels, patch_labels_x, patch_labels_y, patch_labels_z,
                                      x + mx - sx, y + my - sy, z + mz - sz)

        yield patch_features, patch_labels

def randomPatchsAugmented(in_dataset, gd_dataset, patch_number, patch_in_size, patch_gd_size):
    patchs_in = np.zeros((patch_number, patch_in_size[0], patch_in_size[1], patch_in_size[2]), dtype=in_dataset.dtype)
    patchs_gd = np.zeros((patch_number, patch_gd_size[0], patch_gd_size[1], patch_gd_size[2]), dtype=gd_dataset.dtype)

    if(patch_in_size[0] % patch_gd_size[0] != 0 or patch_in_size[1] % patch_gd_size[1] != 0 or patch_in_size[2] % patch_gd_size[2] != 0):
        sys.exit("ERROR : randomPatchsAugmented patchs size error 1")

    if(patch_in_size[0] < patch_gd_size[0] or patch_in_size[1] < patch_gd_size[1] or patch_in_size[2] < patch_gd_size[2]):
        sys.exit("ERROR : randomPatchsAugmented patchs size error 2")

    # middle of patch
    mx = int(patch_in_size[0] / 2)
    my = int(patch_in_size[1] / 2)
    mz = int(patch_in_size[2] / 2)
    # patch label size/2
    sx = int(patch_gd_size[0] / 2)
    sy = int(patch_gd_size[1] / 2)
    sz = int(patch_gd_size[2] / 2)

    for count in range(patch_number):
        id = randint(0, in_dataset.shape[0]-1)
        x  = randint(0, in_dataset.shape[1]-patch_in_size[0])
        y  = randint(0, in_dataset.shape[2]-patch_in_size[1])
        z  = randint(0, in_dataset.shape[3]-patch_in_size[2])

        r0 = randint(0, 3)
        r1 = randint(0, 3)
        r2 = randint(0, 3)

        patchs_in[count] = extractPatch(in_dataset[id], patch_in_size[0], patch_in_size[1], patch_in_size[2], x, y, z)
        patchs_gd[count] = extractPatch(gd_dataset[id], patch_gd_size[0], patch_gd_size[1], patch_gd_size[2], x + mx - sx, y + my - sy, z + mz - sz)

        patchs_in[count] = np.rot90(patchs_in[count], r0, (0, 1))
        patchs_in[count] = np.rot90(patchs_in[count], r1, (1, 2))
        patchs_in[count] = np.rot90(patchs_in[count], r2, (2, 0))

        patchs_gd[count] = np.rot90(patchs_gd[count], r0, (0, 1))
        patchs_gd[count] = np.rot90(patchs_gd[count], r1, (1, 2))
        patchs_gd[count] = np.rot90(patchs_gd[count], r2, (2, 0))

    return patchs_in.reshape(patchs_in.shape[0], patchs_in.shape[1], patchs_in.shape[2], patchs_in.shape[3], 1),\
           patchs_gd.reshape(patchs_gd.shape[0], patchs_gd.shape[1], patchs_gd.shape[2], patchs_gd.shape[3], 1)

def generatorRandomPatchsAugmented(in_dataset, gd_dataset, patch_number, patch_in_size, patch_gd_size):
    while True:
        yield randomPatchsAugmented(in_dataset, gd_dataset, patch_number, patch_in_size, patch_gd_size)
