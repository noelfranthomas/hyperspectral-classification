# This keeps track of general params that will be used in the python batch files
# Includes things like:
# - paths
# - image params
# - ML params, like train rate and number of epochs

import numpy as np
import os

class DefaultParams:
    def __init__(self):
        #self.root_path = "/Users/julianstys/Documents/ML/2021_Oct_8/ML/MasterLabDB59803_PBMC_binary_export_for_ML_fullSpectra/"
        self.root_path = "/home/"+os.environ['USER']+"/projects/def-stys/shared/" # root path to where the data is, by default the shared path

        self.train_rate = 0.75
        self.img_width = 1024
        self.img_height = 1024
        self.num_epochs = 2000
        self.num_batch = 5

        # np.float32 or np.float16
        self.np_precision = np.float16 # float32 is overkill, float16 is enoguh

        # what should the image be multiplied as before training? (useful for normalization)
        self.img_mult_amt = 1/4000

        self.train_subdirs = [["PBMC_DATA_JUL_2021/CSFpos_MCI/tiff", "PBMC_DATA_JUL_2021/CSFpos_AD/tiff"], ["PBMC_DATA_JUL_2021/CSFneg_healthy/tiff"]] # [["CSFpos_MCI", "CSFpos_AD"], ["CSFneg_healthy"]]

        # Don't worry about these varaibles...
        self.train_subdirs_voxstuff = [["PBMC_DATA_JUL_2021/CSFpos_MCI", "PBMC_DATA_JUL_2021/CSFpos_AD"], ["PBMC_DATA_JUL_2021/CSFneg_healthy"]] # [["CSFpos_MCI", "CSFpos_AD"], ["CSFneg_healthy"]]
        self.voxel_stuff = True
        self.voxel_img_width = 50
        self.voxel_img_height = 50
        self.should_shuffle_voxels = False
        self.max_image_leniancy = 0.5
        self.should_flatten_arr = False
        self.should_return_class_names = False
        self.should_separate_patients_for_training = False
        self.should_return_metadata = False
