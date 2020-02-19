# ------------------------------------------------------------ #
#
# file : utils/config/read.py
# author : CM
# Read the configuration
#
# ------------------------------------------------------------ #

import configparser


def readConfig(filename):

    # ----- Read the configuration ----
    config = configparser.RawConfigParser()
    config.read_file(open(filename))

    dataset_in_path     = config.get("dataset", "in_path")
    dataset_gd_path     = config.get("dataset", "gd_path")

    dataset_train       = int(config.get("dataset", "train"))
    dataset_valid       = int(config.get("dataset", "valid"))
    dataset_test        = int(config.get("dataset", "test"))


    train_patch_size_x  = int(config.get("train", "patch_size_x"))
    train_patch_size_y  = int(config.get("train", "patch_size_y"))
    train_patch_size_z  = int(config.get("train", "patch_size_z"))

    train_batch_size    = int(config.get("train", "batch_size"))
    train_steps_per_epoch   = int(config.get("train", "steps_per_epoch"))
    train_epochs        = int(config.get("train", "epochs"))
    
    logs_path           = config.get("train", "logs_path")


    return {"dataset_in_path": dataset_in_path,
            "dataset_gd_path": dataset_gd_path,
            "dataset_train": dataset_train,
            "dataset_valid": dataset_valid,
            "dataset_test": dataset_test,
            "train_patch_size_x": train_patch_size_x,
            "train_patch_size_y": train_patch_size_y,
            "train_patch_size_z": train_patch_size_z,
            "train_batch_size": train_batch_size,
            "train_steps_per_epoch": train_steps_per_epoch,
            "train_epochs": train_epochs,
            "logs_path": logs_path
            }

# Old version will be deleted soon
def readConfig_OLD(filename):

    # ----- Read the configuration ----
    config = configparser.RawConfigParser()
    config.read_file(open(filename))

    dataset_train_size      = int(config.get("dataset","train_size"))
    dataset_train_gd_path   = config.get("dataset","train_gd_path")
    dataset_train_mra_path  = config.get("dataset","train_mra_path")

    dataset_valid_size      = int(config.get("dataset","valid_size"))
    dataset_valid_gd_path   = config.get("dataset","valid_gd_path")
    dataset_valid_mra_path  = config.get("dataset","valid_mra_path")

    dataset_test_size       = int(config.get("dataset","test_size"))
    dataset_test_gd_path    = config.get("dataset","test_gd_path")
    dataset_test_mra_path   = config.get("dataset","test_mra_path")

    image_size_x = int(config.get("data","image_size_x"))
    image_size_y = int(config.get("data","image_size_y"))
    image_size_z = int(config.get("data","image_size_z"))

    patch_size_x = int(config.get("patchs","patch_size_x"))
    patch_size_y = int(config.get("patchs","patch_size_y"))
    patch_size_z = int(config.get("patchs","patch_size_z"))

    batch_size      = int(config.get("train","batch_size"))
    steps_per_epoch = int(config.get("train","steps_per_epoch"))
    epochs          = int(config.get("train","epochs"))

    logs_folder     = config.get("logs","folder")

    return {"dataset_train_size"    : dataset_train_size,
            "dataset_train_gd_path" : dataset_train_gd_path,
            "dataset_train_mra_path": dataset_train_mra_path,
            "dataset_valid_size"    : dataset_valid_size,
            "dataset_valid_gd_path" : dataset_valid_gd_path,
            "dataset_valid_mra_path": dataset_valid_mra_path,
            "dataset_test_size"     : dataset_test_size,
            "dataset_test_gd_path"  : dataset_test_gd_path,
            "dataset_test_mra_path" : dataset_test_mra_path,
            "image_size_x" : image_size_x,
            "image_size_y" : image_size_y,
            "image_size_z" : image_size_z,
            "patch_size_x" : patch_size_x,
            "patch_size_y" : patch_size_y,
            "patch_size_z" : patch_size_z,
            "batch_size" : batch_size,
            "steps_per_epoch" : steps_per_epoch,
            "epochs" : epochs,
            "logs_folder" : logs_folder
            }