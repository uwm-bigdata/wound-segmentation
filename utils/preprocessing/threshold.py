# ------------------------------------------------------------ #
#
# file : preprocessing/threshold.py
# author : CM
# Preprocess function for Bullitt dataset
#
# ------------------------------------------------------------ #
import os
import sys
import numpy as np
import nibabel as nib
from utils.io.write import npToNii
from utils.config.read import readConfig

# Get the threshold for preprocessing
def getThreshold(dataset_mra, dataset_gd):
    threshold = dataset_mra.max()

    for i in range(0,len(dataset_mra)):
        mra = dataset_mra[i]
        gd = dataset_gd[i]

        for x in range(0, mra.shape[0]):
            for y in range(0, mra.shape[1]):
                for z in range(0, mra.shape[2]):
                    if(gd[x,y,z] == 1 and mra[x,y,z] < threshold):
                        threshold = mra[x,y,z]

    return threshold

# Apply threshold to an image
def thresholding(data, threshold):
    output = np.copy(data)
    for x in range(0, data.shape[0]):
        for y in range(0, data.shape[1]):
            for z in range(0, data.shape[2]):
                    if data[x,y,z] > threshold:
                        output[x,y,z] = data[x,y,z]
                    else:
                        output[x,y,z] = 0
    return output

config_filename = sys.argv[1]
if(not os.path.isfile(config_filename)):
    sys.exit(1)

config = readConfig(config_filename)

output_folder = sys.argv[2]
if(not os.path.isdir(output_folder)):
    sys.exit(1)

print("Loading training dataset")

train_mra_dataset = np.empty((30, config["image_size_x"], config["image_size_y"], config["image_size_z"]))
i = 0
files = os.listdir(config["dataset_train_mra_path"])
files.sort()

for filename in files:
    if(i>=30):
       break
    print(filename)
    train_mra_dataset[i, :, :, :] = nib.load(os.path.join(config["dataset_train_mra_path"], filename)).get_data()
    i = i + 1


train_gd_dataset = np.empty((30, config["image_size_x"], config["image_size_y"], config["image_size_z"]))
i = 0
files = os.listdir(config["dataset_train_gd_path"])
files.sort()

for filename in files:
    if(i>=30):
       break
    print(filename)
    train_gd_dataset[i, :, :, :] = nib.load(os.path.join(config["dataset_train_gd_path"], filename)).get_data()
    i = i + 1

print("Compute threshold")
threshold = getThreshold(train_mra_dataset, train_gd_dataset)

train_mra_dataset = None
train_gd_dataset = None

print("Apply preprocessing to test image")
files = os.listdir(config["dataset_test_mra_path"])
files.sort()

for filename in files:
    print(filename)
    data = nib.load(os.path.join(config["dataset_test_mra_path"], filename)).get_data()
    print(np.average(data))
    preprocessed = thresholding(data, threshold)
    print(np.average(preprocessed))
    npToNii(preprocessed,os.path.join(output_folder+'/test_Images', 'pre_'+filename))


print("Apply threshold to train image : ", threshold)
files = os.listdir(config["dataset_train_mra_path"])
files.sort()

for filename in files:
    print(filename)
    data = nib.load(os.path.join(config["dataset_train_mra_path"], filename)).get_data()
    print(np.average(data))
    preprocessed = thresholding(data, threshold)
    print(np.average(preprocessed))
    npToNii(preprocessed,os.path.join(output_folder+'/train_Images', 'pre_'+filename))