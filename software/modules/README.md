
Here is where all of the modules I used to do various pre-processing steps on the images (such as loading the tiff files)

# Setup on my system
Here's where I store them on my system and the file structure, for reference (IMPORTANT: Make sure that each one of these files is in the same directory! Otherwise there might be problems when running your scripts): 

/home/$USER/scripts/modules/*
- binary_to_tiff.py
- default_params.py
- import_binary.py
- module_loadML.py


# Summary 
I'll describe each file briefly

## default_params.py
Contains default parameter values used in my ML algorithms. I then reference all of my ML python scripts to this one, so that if I need to make changes, I only need to change this file instead of all individual scripts.

Shouldn't have to change anything in this script to run on current data. Right now, the root_path of all image data points at the **shared/** folder I mentioned before (**/home/$USER/projects/def-stys/shared/**), and it will read tiff files from 3 subdirectories (**PBMC_DATA_JUL_2021/CSFpos_MCI/tiff**, **PBMC_DATA_JUL_2021/CSFpos_AD/tiff**, **PBMC_DATA_JUL_2021/CSFneg_healthy/tiff**), into 2 classes (notice how the length of self.train_subdirs is 2)


## module_loadML.py
The only useful function in your case in this script is **loadTiffFiles**. Check out the comments as to what the parameters of this script do, but in short, this will convert tiff files to numpy arrays. You can check out where I call **loadTiffFiles** inside **2D_CNN/2D_CNN1/main_final_noflow_CNN.py** to get an idea as to how it works. Notice how I pass parameters from **default_params.py** directly to this function.


## import_binary.py
Not important 

## binary_to_tiff.py
Not important
