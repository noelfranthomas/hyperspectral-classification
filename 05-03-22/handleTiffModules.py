import os
import numpy as np
from PIL import Image
import math
import random
import tensorflow as tf

def loadTiffs(class_dir,root_folder, img_mult_amt=1, max_files=-1):
    # Loads .tiff files into usable numpy array
    # X should be (num_files, 32, 1024, 1024)
    # Y should be (num_files, 1), dtype = int32

    # class_dir should look something like:  [[ class1_dir1, class1_dir2 ... ],   [ class2_dir1, class2_dir2 ... ], ...]

    # root_folder is the root folder which contains all of the image data

    # np_precision is the desired floating point precision of the numpy matrices, usually float16 is fine

    # train_rate is what ratio of images should be used for training and testing, for example train_rate=0.75 means 75% for training, 25% for validation

    # max=-1 means to automatically detect how minimum number of tiffs per class, then set each train amount to that (recommended to keep at -1)
    # - this will ensure a balance of trained files on all classes

    tiff_files_per_class = {}
    # Step 1: Get number of tiff files by class, as well as max number of tiff files allowed
    class_num = 0
    for subdir_list in class_dir:
        num_tiff_files_for_class = 0
        for top_dir in subdir_list:
            print(top_dir)
            for tiff_file in os.listdir(root_folder+"/"+top_dir):
                num_tiff_files_for_class += 1
        tiff_files_per_class[class_num] = num_tiff_files_for_class
        class_num += 1

    max_allowed_tiff_files = max_files

    if max_files == -1: # Calculate class with least number of tiffs
        least_tiff_files = -1
        for k,v in tiff_files_per_class.items():
            if v < least_tiff_files or least_tiff_files < 0:
                least_tiff_files = v
        max_allowed_tiff_files = least_tiff_files
    # Step 2: Get nested shuffled array of tiff files by directory.
    # For example, if I have 2 classes, nested array could look something like:
    # [ [class1_file1.tiff, class1_file2.tiff, class1_file3.tiff ],     [class2_file1.tiff, class2_file2.tiff, class2_file3.tiff ] ]
    class_dir_shuffles = []
    for subdir_list in class_dir:
        class_dir_shuffled_temp = []
        #for top_dir in subdir_list:
        class_shuffle = getShuffledTiffs(subdir_list, root_folder, max_allowed_tiff_files)
        #class_dir_shuffled_temp += class_shuffle

        class_dir_shuffles.append(class_shuffle)

    num_classes = len(class_dir)

    print("\nList of tiff files by class:")
    for class_num in range(0, len(class_dir)):
        print("\nClass "+str(class_num)+" files: (number of files is "+str(len(class_dir_shuffles[class_num]))+")")
        for tiff_file in class_dir_shuffles[class_num]:
            print(tiff_file)

    # Step 5: Get shuffle all data, including class label (0/1 class label array)
    all_tiff_files = []
    all_class = []
    all_tiff_files_final = []
    all_class_final = []

    # Convert into flat array of all tiff files, as well as class label array
    for i in range(0,num_classes):
        for file in class_dir_shuffles[i]:
            all_tiff_files.append(file)
            all_class.append(i)

    file_index = [i for i in range(0,len(all_tiff_files))]
    random.shuffle(file_index)

    for i in range(0,len(file_index)):
        all_tiff_files_final.append(all_tiff_files[file_index[i]])
        all_class_final.append(all_class[file_index[i]])

    print("-------------")

    print(file_index)
    print(all_tiff_files_final)
    print(all_class_final)

    return all_tiff_files_final, all_class_final

def getShuffledTiffs(class_subdir, root_dir, max_files):
    # Reads all .tiff files in tiff_dir, and returns a shuffled array of them
    all_files = []
    num_tiff_files_added = 0
    for tiff_dir in class_subdir:
        for file in os.listdir(root_dir+"/"+tiff_dir):
            if ".tiff" in file and num_tiff_files_added < max_files:
                all_files.append(root_dir+"/"+tiff_dir+"/"+file)
                num_tiff_files_added += 1
    
    random.shuffle(all_files)
    
    return all_files