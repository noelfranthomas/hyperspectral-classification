# This script contains the module for loading all the data from a file into a usable numpy array

import os
import time
import numpy as np
from PIL import Image
import math
##from keras.models import load_model
import random
import sys
module_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(module_dir)
import import_binary as imp


def getShuffledTiffs(class_subdir, root_dir, max_files):
    # Reads all .tiff files in tiff_dir, and returns a shuffled array of them
    all_files = []
    for tiff_dir in class_subdir:
        for file in os.listdir(root_dir+"/"+tiff_dir):
            if ".tiff" in file:
                all_files.append(root_dir+"/"+tiff_dir+"/"+file)
    random.shuffle(all_files)
    
    #print("getShuffledTiffs")
    #print(max_files)
    final_tiff_files = []
    num_tiff_files_added = 0
    for tiff_file in all_files:
        if num_tiff_files_added >= max_files:
            break
        final_tiff_files.append(tiff_file)
        num_tiff_files_added += 1
    return final_tiff_files

def saveImagesAsJPEG(images, top_dir, class_vals=[], class_index=[], patient_name_index={}):
    # Saves image as jpeg for viewing
    # images should be of the form (num_files, 32, 1024, 1024)
    # Approximates 32 channels as rgb for viewing
    for i in range(0, images.shape[0]):
        rgb_np = np.zeros((3, images.shape[1], images.shape[2]), dtype=np.float32)
        rgb_np[2,:,:] = images[i,:,:,4]
        rgb_np[1,:,:] = images[i,:,:,14]
        rgb_np[0,:,:] = images[i,:,:,27]
        rgb_np = 255*rgb_np/np.amax(rgb_np)
        rgb_img = Image.fromarray(np.swapaxes(rgb_np.astype('uint8'),0,2), 'RGB')
        if class_vals == []:
            rgb_img.save(top_dir+"/"+str(i)+"_.jpeg")
        else:
            if patient_name_index == {} or class_index == []:
                rgb_img.save(top_dir+"/"+str(i)+"_"+str(class_vals[i])+".jpeg")
            else:
                rgb_img.save(top_dir+"/"+str(i)+"_"+str(class_vals[i])+"_"+patient_name_index[class_index[i]].split('/')[-1]+".jpeg")

def saveImagesAsLayeredJPEG(images, stack_amt):
    # saves as jpeg, but stacks multiple on top of each other
    # useful to see if images are seemless
    # Saves image as jpeg for viewing
    # images should be of the form (num_files, 32, 1024, 1024)
    # Approximates 32 channels as rgb for viewing
    for i in range(0, int(math.floor(images.shape[0] / stack_amt))):
        rgb_np = np.zeros((3, images.shape[2], images.shape[3]*stack_amt), dtype=np.float32)
        for j in range(0, stack_amt):
            rgb_np[2,:,:] = images[i,4,:,:]
            rgb_np[1,:,:] = images[i,14,:,:]
            rgb_np[0,:,:] = images[i,27,:,:]
            rgb_np = 255*rgb_np/np.amax(rgb_np)
            rgb_img = Image.fromarray(np.swapaxes(rgb_np.astype('uint8'),0,2), 'RGB')
        rgb_img.save("/Users/julianstys/Documents/ML/2021_Oct_8/ML/images/"+str(i)+".jpeg")


def getXYForFlatPixel(flat_pix, x_width, y_height):
    # flat_pix is 0 to num_pixels_in_all_frames
    # x_width and y_height are width of images
    # returns (num_image, x, y) given flat pixel

    num_image = int(math.floor(flat_pix / (x_width*y_height) ))

    flat_pix_rem = flat_pix % (x_width*y_height)

    y_coor = math.floor(flat_pix_rem / x_width)
    x_coor = flat_pix_rem % y_height

    return num_image, x_coor, y_coor

def multiShuffleTwoArrays(a, b):
    # Will shuffle arrays a and b using the same shuffled indices
    assert a.shape[0] == b.shape[0]

    indices = np.random.permutation(a.shape[0])
    return a[indices], b[indices]

def multiShuffleTwoArrays3(a, b, c):
    # Will shuffle arrays a and b using the same shuffled indices
    print(a.shape)
    print(b.shape)
    print(c.shape)
    assert a.shape[0] == b.shape[0]
    assert a.shape[0] == c.shape[0]

    indices = np.random.permutation(a.shape[0])
    return a[indices], b[indices], c[indices]


def clampVal(val, min, max):
    if val < min:
        return min
    elif val > max:
        return max
    return val

def getNumImagesToScrapByParams(max_images_per_class, num_pseudoimages_by_class, max_image_leniancy):
    max_images = max_images_per_class
    if max_images_per_class < 0:
        # need to count least class
        least_num_images = -1
        least_class = -1

        for i in range(0, len(num_pseudoimages_by_class)):
            if num_pseudoimages_by_class[i] < least_num_images or least_num_images == -1:
                least_num_images = num_pseudoimages_by_class[i]
                least_class = i
        max_images = least_num_images

    max_images = int(max_images + max_image_leniancy*max_images)

    num_images_to_remove = []

    for i in range(0, len(num_pseudoimages_by_class)):
        num_images_to_remove.append( clampVal( num_pseudoimages_by_class[i] -  max_images, 0, 99999999))
    print("removing images")
    print(num_images_to_remove)
    return num_images_to_remove

def getBinaryFilesForSubdir(root_folder, class_dirs, max_num_files=-1):
    # Returns [ [binary1, binary2, binary3...], [binary1, binary2, binary3...], ...  ] nested for each class
    binary_files = []

    for subdir_list in class_dirs:
        binary_files_temp = []
        num_files_in_class = 0
        for subdir in subdir_list:
            print(subdir)
            for file in os.listdir(root_folder+"/"+subdir):
                if ".itk_binary" in file:
                    if num_files_in_class < max_num_files or max_num_files == -1:
                        binary_files_temp.append(root_folder+"/"+subdir+"/"+file)
                        num_files_in_class += 1

        binary_files.append(binary_files_temp)

    return binary_files

def readAllBinaryHeaderFiles(all_binary_files):
    all_header_info = {} # key is filename, val is list of: [classnum, nX, nY, nZ, total_pixels]
    class_num = 0
    for class_files in all_binary_files:
        for file in class_files:
            filename = file.split('/')[-1]
            binary_data = imp.import_binary(file, filename, voxel_stuff=False, shuffle_voxels=False, only_read_header=True)
            binary_data.print_all_data()
            all_header_info[filename] = [class_num, binary_data.get_nx(), binary_data.get_ny(), binary_data.get_nz(), binary_data.get_totalpixels(), binary_data.get_numfloats()]
            #exit()
            #saveImagesAsJPEG(binary_data, "C:/Users/julia/Documents/ML/images", class_vals=[])
        class_num += 1
    return all_header_info

def getNumImagesForPatient(header_value, voxel_stuff, x_width, y_height):
    # header_value is [class_num, nx, ny, nz, total_pixels]
    if voxel_stuff:
        # Need to calculate from x_width, y_height and total_pixels
        return math.floor(float(header_value[4]) / float(x_width * y_height))
    else:
        # Read directly from header_value
        return header_value[3]

def getShapeOfArrays(all_header_info, voxel_stuff, x_width, y_height, max_image_leniancy, train_rate, should_separate_patients_for_training):
    # return: array_index_dict_all, num_train_iamges, num_val_images

    # Step 1: Find total num images per class
    # will also append num image to end of all_header_info
    num_images_by_class = {} # k = class_int, v = num images for class
    for k, v in all_header_info.items():
        # v is [class_num, nx, ny, nz, total_pixels]
        num_images_for_class = getNumImagesForPatient(v, voxel_stuff, x_width, y_height)
        all_header_info[k].append(num_images_for_class)

        if v[0] in num_images_by_class:
            num_images_by_class[v[0]] += num_images_for_class
        else:
            num_images_by_class[v[0]] = num_images_for_class


    print("\nnum_images_by_class")
    for k,v in num_images_by_class.items():
        print(k,v)

    # Step 2: Find class with least number of images in it
    # also get max number of images allowed
    class_with_least_images = -1
    least_num_images = -1

    for k, v in num_images_by_class.items():
        if v < least_num_images or least_num_images < 0:
            class_with_least_images = k
            least_num_images = v

    # maximum number of images allowed
    max_number_of_images_allowed = int(math.floor(float(least_num_images) + float(least_num_images) * float(max_image_leniancy)))

    print("class_with_least_images "+str(class_with_least_images))
    print("max_number_of_images_allowed "+str(max_number_of_images_allowed))


    # Step 3: Get array of image names, and shuffle it
    image_arr = []
    for k, v in all_header_info.items():
        image_arr.append(k)
    random.shuffle(image_arr)
    print(image_arr)

    # Step 4: Initialize image increment dictionary
    # This will increment up and when exceeded, will stop reading files
    image_inc_dict = {}
    for k, v in num_images_by_class.items():
        image_inc_dict[k] = 0

    # Step 5: Based on shuffled image names, get dictionary of: key = filename, value = [unshuffled index of current filename]
    # array_index_dict: key = image name, val =
    num_images_per_class = {} # key = class_num, val = num images by class
    array_index_dict = {} # key = image name, val = [shuffling index of where to place files for reading]
    total_num_images_per_class = {}
    total_num_images = 0
    total_num_train_images = 0
    total_num_val_images = 0
    base_image_num = 0
    for image in image_arr:

        class_num = all_header_info[image][0]
        if class_num not in num_images_per_class:
            num_images_per_class[class_num] = 0


        # append index
        image_index_for_class = []
        for i in range(base_image_num, base_image_num+all_header_info[image][-1]):
            image_inc_dict[class_num] += 1
            if image_inc_dict[class_num] < max_number_of_images_allowed:
                #if i < max_number_of_images:
                image_index_for_class.append(i)
                if class_num not in total_num_images_per_class:
                    total_num_images_per_class[class_num] = 0
                total_num_images_per_class[class_num] += 1
                base_image_num += 1
                num_images_per_class[class_num] += 1

        if len(image_index_for_class) > 4 or True: # Threshold, don't want a very low number of images for trainign for a single class
            array_index_dict[image] = image_index_for_class

    for k,v in total_num_images_per_class.items():
        total_num_images += v

    print("total_num_images_per_class")
    print(total_num_images_per_class)
    print("total_num_images : "+str(total_num_images))
    total_num_train_images = int(math.floor(total_num_images * train_rate))
    total_num_val_images = total_num_images - total_num_train_images
    print("total_num_train_images : "+str(total_num_train_images))
    print("total_num_val_images : "+str(total_num_val_images))

    # Step 6: Now we need to do one of two things:
    # Separate by patient
    # No separation
    # This variable will be a shuffled index of where to place files
    image_shuffled_index = []

    array_index_dict_all = {} # is the same as array_index_dict, but with random index values instead. Train and val dicts will be built off of this

    train_or_val = {}
    if should_separate_patients_for_training: # Should separate images for training
        # Firstly, iterate through and get new number of images to train and shuffle
        # This will change slightly due to the separation by patient requirement
        # Will add to training, instead of validation
        for k,v in array_index_dict.items():
            image_is_train = False
            for index in v:
                if index >= total_num_train_images:
                    # Ideally, would add to val. However check if there's already images as part of the train set
                    if image_is_train:
                        total_num_train_images += 1
                else:
                    image_is_train = True

        total_num_val_images = total_num_images - total_num_train_images
        print("new number of train iamges: "+ str(total_num_train_images))
        print("new number of val images: "+str(total_num_val_images))
        image_train_shuffled_index = [i for i in range(0, total_num_train_images)]
        image_val_shuffled_index = [i for i in range(0, total_num_val_images)]

        random.shuffle(image_train_shuffled_index)
        random.shuffle(image_val_shuffled_index)


        # Build random shuffled index:
        for k,v in array_index_dict.items():
            for index in v:
                if k not in array_index_dict_all:
                    array_index_dict_all[k] = []
                if index >= total_num_train_images:
                    # Add via val index
                    if k not in train_or_val:
                        train_or_val[k] = "val"
                    val_index = image_val_shuffled_index[index - total_num_train_images] + total_num_train_images
                    array_index_dict_all[k].append(val_index)

                else:
                    # Add via train index
                    if k not in train_or_val:
                        train_or_val[k] = "train"

                    train_index = image_train_shuffled_index[index]
                    array_index_dict_all[k].append(train_index)


        #exit()
    else: # Should NOT separate iamges for training
        # Make image_shuffled_index a flat shuffled index of len all images
        image_shuffled_index = [i for i in range(0, total_num_images)]
        random.shuffle(image_shuffled_index)

        # Build random shuffled index:
        for k,v in array_index_dict.items():
            for index in v:
                if k not in array_index_dict_all:
                    array_index_dict_all[k] = []
                array_index_dict_all[k].append(image_shuffled_index[index])


    print("checker: ")
    all_checked = {}
    for k,v in array_index_dict_all.items():
        for index in v:
            if index in all_checked:
                print("ERROR: duplicate ")
                print(index)
                print(k)
                print(v)
            elif index >= total_num_images:
                print("ERROR: index > num images")
                print(index)
                print(k)
                print(v)
            all_checked[index] = True
            #all_checked.append(index)
    print("check 1")
    for i in range(0, total_num_images):
        if i not in all_checked:
            print("index not in all checked:  "+str(i))
    print("check 2")
    for k,v in array_index_dict_all.items():
        if k in train_or_val:
            for index in v:
                if train_or_val[k] == "train" and index >= total_num_train_images:
                    print("ERROR: badly placed train")
                    print(k)
                    print(v)
                    print(index)
                elif train_or_val[k] == "val" and index < total_num_train_images:
                    print("ERROR: badly placed val")
                    print(k)
                    print(v)
                    print(index)
    print("check 3")
    print("finished check")

    return array_index_dict_all, total_num_train_images, total_num_val_images


def getShapeOfArrays2(all_header_info, voxel_stuff, x_width, y_height, max_image_leniancy, train_rate, should_separate_patients_for_training):
    # Returns:
    # "train_shape" : shape of train array
    # "val_shape" : shape of val array
    # "train_index" : index of shuffled arrays that can be used for
    # "val_index"
    array_shape_dict = {}


    # Step 1: Find total num images per class
    # will also append num image to end of all_header_info
    num_images_by_class = {}
    for k, v in all_header_info.items():
        # v is [class_num, nx, ny, nz, total_pixels]
        num_images_for_class = getNumImagesForPatient(v, voxel_stuff, x_width, y_height)
        all_header_info[k].append(num_images_for_class)

        if v[0] in num_images_by_class:
            num_images_by_class[v[0]] += num_images_for_class
        else:
            num_images_by_class[v[0]] = num_images_for_class

    # Step 2: Find class with least number of images in it
    # also get max number of images allowed
    class_with_least_images = -1
    least_num_images = -1

    for k, v in num_images_by_class.items():
        if v < least_num_images or least_num_images < 0:
            class_with_least_images = k
            least_num_images = v

    max_number_of_images = int(math.floor(float(least_num_images) + float(least_num_images) * float(max_image_leniancy)))
    #print("max images "+str(max_number_of_images))
    # Step 3: Get array of image names, and shuffle it
    image_arr = []
    for k, v in all_header_info.items():
        image_arr.append(k)
    random.shuffle(image_arr)

    print(image_arr)

    # Step 4: Initialize image increment dictionary
    # This will increment up and when exceeded, will stop reading files
    image_inc_dict = {}
    for k, v in num_images_by_class.items():
        image_inc_dict[k] = 0


    # Step 5:
    # array_index_dict: key = image name, val =
    num_images_per_class = {} # key = class_num, val = num images by class
    array_index_dict = {} # key = image name, val = [shuffling index of where to place files for reading]
    total_num_images_per_class = {}
    base_image_num = 0
    base_image_num_by_class = 0
    for image in image_arr:

        class_num = all_header_info[image][0]
        if class_num not in num_images_per_class:
            num_images_per_class[class_num] = 0


        #base_image_num = num_images_per_class[class_num]
        base_image_num_by_class = num_images_per_class[class_num]

        # append index
        image_index_for_class = []
        for i in range(base_image_num, base_image_num+all_header_info[image][-1]):
            image_inc_dict[class_num] += 1
            if image_inc_dict[class_num] < max_number_of_images:
                #if i < max_number_of_images:
                image_index_for_class.append(i)
                if class_num not in total_num_images_per_class:
                    total_num_images_per_class[class_num] = 0
                total_num_images_per_class[class_num] += 1
                base_image_num += 1
                num_images_per_class[class_num] += 1

        if len(image_index_for_class) > 2: # Threshold, don't want a very low number of images for trainign for a single class
            ##random.shuffle(image_index_for_class)
            array_index_dict[image] = image_index_for_class

        #base_image_num += all_header_info[image][-1]
        ##num_images_per_class[class_num] += all_header_info[image][-1] # append number of images to num_images_per_class

    #print("array_index_dict")
    print("total_num_images_per_class")
    print(total_num_images_per_class)

    total_images_of_all_classes = 0
    for k,v in total_num_images_per_class.items():
        total_images_of_all_classes += v


    train_images_by_class = {}
    # Step 6: Build long list of shuffed inde of all_num_images
    all_num_images = base_image_num
    all_shuffled_index = [i for i in range(0, all_num_images)]
    random.shuffle(all_shuffled_index)
    array_rand_index_dict = {}

    print("array_index_dict")
    for k,v in array_index_dict.items():
        class_num = all_header_info[k][0]
        print(class_num, k, v)

    train_images_by_class = {}
    val_images_by_class = {}

    total_train_index = 0
    total_val_index = 0

    total_train_images = 0
    total_val_images = 0
    num_train_image_for_class = {}
    train_inc_by_class = {}

    # build train and val dicts
    array_train_index_dict = {}
    array_val_index_dict = {}


    for k,v in total_num_images_per_class.items():
        num_train_images = math.floor( train_rate * v )
        num_val_images = v - num_train_images
        total_train_images += num_train_images
        total_val_images += num_val_images
        num_train_image_for_class[k] = num_train_images
        train_inc_by_class[k] = 0

    if should_separate_patients_for_training :
        # num_train_images should be different for each class

        # inc
        num_train_images_added_for_class = {}
        num_val_images_added_for_class = {}
        # should be added to train or val
        has_finished_adding_train_by_class = {}
        max_image_by_class = {}
        for k, v in total_num_images_per_class.items():
            has_finished_adding_train_by_class[k] = False
            num_train_images_added_for_class[k] = 0
            num_val_images_added_for_class[k] = 0
            max_image_by_class[k] = -1

        total_train_images_for_all_classes = 0
        # total_num_images_per_class


        for class_int, num_image in total_num_images_per_class.items():
            num_train_image_for_class = math.floor(num_image * train_rate)
            num_val_image_for_class = num_image - num_train_image_for_class
            print("iterationg")
            print(class_int, num_image)
            for image in image_arr:
                class_num = all_header_info[image][0]

                if class_num == class_int:
                    for image_name, index in array_index_dict.items():
                        if image_name == image:
                            if not has_finished_adding_train_by_class[class_num]:
                                # add to training
                                train_images_by_class[image] = index
                                num_train_images_added_for_class[class_num] += len(index)
                                total_train_images_for_all_classes += len(index)
                                if index[-1] > max_image_by_class[class_num]:
                                    max_image_by_class[class_num] = index[-1]
                            else:
                                index_temp = []
                                for i in index:
                                    index_temp.append(i)
                                val_images_by_class[image] = index_temp
                                num_val_images_added_for_class[class_num] += len(index)

                    if num_train_images_added_for_class[class_num] > num_train_image_for_class:
                        #
                        has_finished_adding_train_by_class[class_num] = True
        val_subtract_amt = -1
        for k,v in val_images_by_class.items():
            if v[0] < val_subtract_amt or val_subtract_amt < 0:
                val_subtract_amt = v[0]


        # shrink both arrays
        for k,v in train_images_by_class.items():
            temp_train_arr = []
            for i in v:
                temp_train_arr.append(total_train_index)
                total_train_index += 1
            train_images_by_class[k] = temp_train_arr

        for k,v in val_images_by_class.items():
            temp_val_arr = []
            for i in v:
                temp_val_arr.append(total_val_index)
                total_val_index += 1
            val_images_by_class[k] = temp_val_arr

        print("\nnum_train_images_added_for_class")
        print(num_train_images_added_for_class)
        print("num_val_images_added_for_class")
        print(num_val_images_added_for_class)
        print("train_images_by_class")
        for k,v in train_images_by_class.items():
            print(all_header_info[k][0],k,v)
        print("\nval_images_by_class")
        for k,v in val_images_by_class.items():
            print(all_header_info[k][0],k,v)
        print("total_num_images_per_class")
        print(total_num_images_per_class)
        print("split")
        print("for class 0 :"+str(float(num_val_images_added_for_class[0])/(float(num_val_images_added_for_class[0])+float(num_train_images_added_for_class[0]))))

        print("for class 1 :"+str(float(num_val_images_added_for_class[1])/(float(num_val_images_added_for_class[1])+float(num_train_images_added_for_class[1]))))
        #total_train_images
        #total_val_images
        total_train_images = 0
        total_val_images = 0
        for k,v in num_val_images_added_for_class.items():
            total_val_images += v
        for k,v in num_train_images_added_for_class.items():
            total_train_images += v

        shuffled_train_index = [i for i in range(0, total_train_images)]
        shuffled_val_index = [i for i in range(0, total_val_images)]
        random.shuffle(shuffled_train_index)
        random.shuffle(shuffled_val_index)

        print(shuffled_train_index)
        print(shuffled_val_index)
        for image_name, index in train_images_by_class.items():
            for i in index:
                if image_name not in array_train_index_dict:
                    array_train_index_dict[image_name] = []
                array_train_index_dict[image_name].append(shuffled_train_index[i])

        for image_name, index in val_images_by_class.items():
            for i in index:
                if image_name not in array_val_index_dict:
                    array_val_index_dict[image_name] = []
                array_val_index_dict[image_name].append(shuffled_val_index[i])
        # Random shuffle per class
        #train_images_by_class
        #val_images_by_class
        #array_train_index_dict
        #array_val_index_dict
        print("\nfinal train")
        for k,v in array_train_index_dict.items():
            print(all_header_info[k][0], k, v)
        print("\nfinal val")
        for k,v in array_val_index_dict.items():
            print(all_header_info[k][0], k, v)

    else:
        print("-------------------")
        #train_images_by_class
        #val_images_by_class

        # all images
        print(all_num_images)
        print(all_shuffled_index)
        print(total_num_images_per_class)


        print("num_train_image_for_class")
        print(num_train_image_for_class)
        print("shuffled index")
        print(all_shuffled_index)
        # Algorithm:
        # have shuffled

        for image_name, index in array_index_dict.items():
            class_num = all_header_info[image_name][0]
            print(class_num, image_name, index)

        shuffled_train_index = [i for i in range(0, total_train_images)]
        shuffled_val_index = [i for i in range(0, total_val_images)]


        array_index_dict_shuffled = {}
        for image_name, index in array_index_dict.items():
            if image_name not in array_index_dict_shuffled:
                array_index_dict_shuffled[image_name] = []
            for i in index:
                array_index_dict_shuffled[image_name].append(all_shuffled_index[i])
        print("\n\nshuffled arr:")
        for k,v in array_index_dict_shuffled.items():
            print(all_header_info[k][0], k, v)




        for image_name, index in array_index_dict_shuffled.items():
            for i in index:
                if i < total_train_images or True:  #num_train_image_for_class[all_header_info[image_name][0]]:
                    if image_name not in array_train_index_dict:
                        array_train_index_dict[image_name] = []
                    array_train_index_dict[image_name].append(i)
                else:
                    if image_name not in array_val_index_dict:
                        array_val_index_dict[image_name] = []
                    array_val_index_dict[image_name].append(i - total_train_images)

        print("\n\n\ntrain")
        num_train = 0
        for k,v in array_train_index_dict.items():
            num_train += len(v)
            print(all_header_info[k][0], k, v)
        print("\n\n\nval")
        num_val = 0
        for k,v in array_val_index_dict.items():
            num_val += len(v)
            print(all_header_info[k][0], k, v)
        print(num_train)
        print(num_val)

        # Algorithm:
        # have shuffled train and val indices
        #


        '''train_class_inc_by_class = {}
        for k,v in total_num_images_per_class.items():
            train_class_inc_by_class[k] = 0
        for class_int, num_image in total_num_images_per_class.items():
            num_train_image_for_class = math.floor(num_image * train_rate)
            num_val_image_for_class = num_image - num_train_image_for_class
            print("-----")
            train_class_inc = 0
            for image_name,index in array_index_dict.items():

                class_num = all_header_info[image_name][0]
                if class_int == class_num:
                    #train_images_by_class
                    for i in index:
                        if train_class_inc_by_class[class_num] < num_train_image_for_class:
                            if image_name not in train_images_by_class:
                                train_images_by_class[image_name] = []
                            train_images_by_class[image_name].append(0)
                            train_class_inc_by_class[class_num] += 1
                        else:
                            if image_name not in val_images_by_class:
                                val_images_by_class[image_name] = []
                            val_images_by_class[image_name].append(0)
                    print(class_num, image_name, index)'''


        # shrink
        '''for k,v in train_images_by_class.items():
            temp_train_arr = []
            for i in v:
                temp_train_arr.append(total_train_index)
                total_train_index += 1
            train_images_by_class[k] = temp_train_arr

        for k,v in val_images_by_class.items():
            temp_val_arr = []
            for i in v:
                temp_val_arr.append(total_val_index)
                total_val_index += 1
            val_images_by_class[k] = temp_val_arr'''


    return array_train_index_dict, array_val_index_dict, total_train_images, total_val_images


# Loads data from .itk_binary file
# Unless specifically using this file format, then use loadTiffFiles instead
def loadBinaryFiles(class_dir,root_folder, train_rate, np_precision, img_mult_amt, max_image_leniancy=0.0,  voxel_stuff=False, x_width=-1, y_height=-1, should_shuffle_pseudoimages=True, should_shuffle_voxels=False, should_use_cutoff_img=True, should_flatten_arr=False, should_return_class_names=False, should_separate_patients_for_training=False, should_return_metadata=False):
    # Step 1: Get all binary files in each subdir
    all_binary_files = getBinaryFilesForSubdir(root_folder, class_dir, max_num_files=-1)

    # Step 2: Read through all metadata from all binary files to set up arrays ahead of time
    all_header_info = readAllBinaryHeaderFiles(all_binary_files)

    # If not voxel stuffing, then set width and height based on image width and height
    num_floats = 0
    for k,v in all_header_info.items():
        if not voxel_stuff:
            x_width = v[1]
            y_height = v[2]
        num_floats = v[5]
    # Get number of header files

    # Step 3: Given metadata info, find size of arrays needed to store all data ahead of time
    array_index_dict_all, num_train_images, num_val_images = getShapeOfArrays(all_header_info, voxel_stuff, x_width, y_height, max_image_leniancy, train_rate, should_separate_patients_for_training)

    print("\n\nreturn from getShapeOfArrays")
    #for k,v in array_index_dict_all.items():
    #    print(k,v)
    print("num train iamges "+str(num_train_images))
    print("num val images: "+str(num_val_images))

    # Step 4: Build empty train and val array
    X_train_all = np.zeros((num_train_images, x_width, y_height, 32), dtype=np_precision)
    X_val_all = np.zeros((num_val_images, x_width, y_height, 32), dtype=np_precision)
    Y_train_all = np.zeros((num_train_images), dtype=np_precision)
    Y_val_all = np.zeros((num_val_images), dtype=np_precision)
    Y_train_class_index = np.zeros((num_train_images), dtype=np_precision) #
    Y_val_class_index = np.zeros((num_val_images), dtype=np_precision) #
    Y_train_metadata_all = np.zeros((num_train_images, num_floats), dtype=np.float32) # keeps track of all metadata, 1 array for each class
    Y_val_metadata_all = np.zeros((num_val_images, num_floats), dtype=np.float32) # keeps track of all metadata, 1 array for each class

    patient_name_index = {} # k = patient_file_index (unique for each binary file), v = filename

    print("Shape of every array:")
    print("X_train_all")
    print(X_train_all.shape)
    print("X_val_all")
    print(X_val_all.shape)
    print("Y_train_all")
    print(Y_train_all.shape)
    print("Y_val_all")
    print(Y_val_all.shape)
    print("Y_train_class_index")
    print(Y_train_class_index.shape)
    print("Y_val_class_index")
    print(Y_val_class_index.shape)




    # Step 5: Stuff array with data from all files
    # integer for each CLASS
    class_num = -1

    # integer for each BINARY FILE
    class_index = -1

    for class_list in all_binary_files:
        class_num += 1
        for file in class_list:

            filename = file.split('/')[-1]
            if filename in array_index_dict_all:
                class_index += 1
                # Need to read from binary file
                patient_name_index[class_index] = file
                print("\nReading file: "+file)
                img_data = imp.import_binary(file, file.split('/')[-1], voxel_stuff=voxel_stuff, shuffle_voxels=should_shuffle_voxels)
                # Iterate through all pixel values
                num_image_for_class = len(array_index_dict_all[filename])
                prev_num_image = 0

                # For voxel stuffing
                print(img_data.get_imgdata().shape)
                # if voxel stuffing, shape will be somthing like (1, NUMPIXELS, 32)
                # else, will be (32, 1024, 1024, NUMIMAGES)
                if voxel_stuff:
                    for pix_num in range(0, img_data.get_imgdata().shape[1]):
                        num_image, x, y = getXYForFlatPixel(pix_num, x_width, y_height)
                        if prev_num_image != num_image:
                            prev_num_image = num_image
                            if num_image >= num_image_for_class:
                                print("finished for class "+filename)

                                break
                        num_image_index = array_index_dict_all[filename][num_image]

                        # Now add to either train or val:
                        if num_image_index < num_train_images:
                            if np.count_nonzero(X_train_all[num_image_index, x, y, :]) > 0:
                                print("WARNING: OVERRIDE (train)")
                            # There will be a bunch of overriting for Y_train_all, Y_val_all, Y_train_class_index etc., but it shouldn't matter in this case (just slightly slower loading times)
                            X_train_all[num_image_index, x, y, :] = img_data.get_imgdata()[0,pix_num, :]*img_mult_amt
                            Y_train_all[num_image_index] = class_num
                            Y_train_class_index[num_image_index] = class_index

                            # Append metadata info
                            for i in range(0, len(img_data.get_metadata())):
                                Y_train_metadata_all[num_image_index, i] = img_data.get_metadata()[i]
                        else:
                            if np.count_nonzero(X_val_all[num_image_index-num_train_images, x, y, :]) > 0:
                                print("WARNING: OVERRIDE (val)")
                            if num_image_index-num_train_images < 0 or num_image_index-num_train_images > X_val_all.shape[0]:
                                print("warning, out of range: "+str(num_image_index), "num train: "+str(num_train_images))
                            X_val_all[num_image_index-num_train_images, x, y, :] = img_data.get_imgdata()[0,pix_num, :]*img_mult_amt
                            Y_val_all[num_image_index-num_train_images] = class_num
                            Y_val_class_index[num_image_index-num_train_images] = class_index
                            for i in range(0, len(img_data.get_metadata())):
                                Y_val_metadata_all[num_image_index-num_train_images, i] = img_data.get_metadata()[i]
                    #print(Y_val_metadata_all[0, num_image_index, :])
                else: # size is (32, 1024, 1024, NUMIMAGES)
                    if num_image_for_class > 0: #
                        for num_image in range(0, num_image_for_class): # Reltative to each file, so starts at 0
                            num_image_index = array_index_dict_all[filename][num_image]
                            if num_image_index < num_train_images: # Add to train
                                if np.count_nonzero(X_train_all[num_image_index, :, :, :]) > 0:
                                    print("WARNING: OVERRIDE (train)")
                                X_train_all[num_image_index, :,:,:] = img_data.get_imgdata()[:,:,:,num_image]*img_mult_amt
                                Y_train_all[num_image_index] = class_num
                                Y_train_class_index[num_image_index] = class_index

                                # Append metadata info
                                for i in range(0, len(img_data.get_metadata())):
                                    Y_train_metadata_all[num_image_index, i] = img_data.get_metadata()[i]
                            else: # Add to val
                                if np.count_nonzero(X_val_all[num_image_index-num_train_images, :, :, :]) > 0:
                                    print("WARNING: OVERRIDE (val)")
                                X_val_all[num_image_index-num_train_images, :,:,:] = img_data.get_imgdata()[:,:,:,num_image]*img_mult_amt
                                Y_val_all[num_image_index-num_train_images] = class_num
                                Y_val_class_index[num_image_index-num_train_images] = class_index
                                for i in range(0, len(img_data.get_metadata())):
                                    Y_val_metadata_all[num_image_index-num_train_images, i] = img_data.get_metadata()[i]
                #exit()
    print("Finished reading all binary files")
    print(X_train_all.shape)
    print(X_val_all.shape)

    print("final checker before return:")
    for i in range(0,X_train_all.shape[0]):
        if np.count_nonzero(X_train_all[i,:,:,:]) == 0:
            print("WARNING: 0 found in X_train_all at index")
            print(i)

    for i in range(0,X_val_all.shape[0]):
        if np.count_nonzero(X_val_all[i,:,:,:]) == 0:
            print("WARNING: 0 found in X_val_all at index")
            print(i)

    for i in range(0,Y_train_metadata_all.shape[0]):
        if np.count_nonzero(Y_train_metadata_all[i,:]) == 0:
            print("WARNING: 0 found in Y_train_metadata_all at index")
            print(i)

    for i in range(0,Y_val_metadata_all.shape[0]):
        if np.count_nonzero(Y_val_metadata_all[i,:]) == 0:
            print("WARNING: 0 found in Y_val_metadata_all at index")
            print(i)

    print("\nShape of every array:")
    print("X_train_all")
    print(X_train_all.shape)
    print("X_val_all")
    print(X_val_all.shape)
    print("Y_train_all")
    print(Y_train_all.shape)
    print("Y_val_all")
    print(Y_val_all.shape)
    print("Y_train_class_index")
    print(Y_train_class_index.shape)
    print("Y_val_class_index")
    print(Y_val_class_index.shape)

    # If needed, flatten array here
    if should_flatten_arr:
        X_train_all = np.reshape(X_train_all, (X_train_all.shape[0], X_train_all.shape[1]*X_train_all.shape[2], X_train_all.shape[3]))
        X_val_all = np.reshape(X_val_all, (X_val_all.shape[0], X_val_all.shape[1]*X_val_all.shape[2], X_val_all.shape[3]))

        print("\nShape of every array after flattening:")
        print("X_train_all")
        print(X_train_all.shape)
        print("X_val_all")
        print(X_val_all.shape)
        print("Y_train_all")
        print(Y_train_all.shape)
        print("Y_val_all")
        print(Y_val_all.shape)
        print("Y_train_class_index")
        print(Y_train_class_index.shape)
        print("Y_val_class_index")
        print(Y_val_class_index.shape)

    if should_return_class_names:
        if should_return_metadata:
            return  X_train_all, Y_train_all, X_val_all, Y_val_all, Y_train_class_index, Y_val_class_index, Y_train_metadata_all, Y_val_metadata_all, patient_name_index,
        else:
            return  X_train_all, Y_train_all, X_val_all, Y_val_all, Y_train_class_index, Y_val_class_index, patient_name_index

    if should_return_metadata:
        return  X_train_all, Y_train_all, X_val_all, Y_val_all, Y_train_metadata_all, Y_val_metadata_all
    else:
        return  X_train_all, Y_train_all, X_val_all, Y_val_all



# Loads .tiff files directly
# Recommended using this one instead of loadBinaryFiles
def loadTiffFiles(class_dir,root_folder, train_rate, np_precision, img_mult_amt=1, max_files=-1):
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
    print("tiff_files_per_class: ")
    print(tiff_files_per_class)
    print("max_allowed_tiff_files: ")
    print(max_allowed_tiff_files)

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

    # Step 3: Get number of images in total, and then calcualte number of train images and validation images 
    total_num_images = 0
    for class_num in range(0, len(class_dir)):
        total_num_images += len(class_dir_shuffles[class_num])
    num_train = math.floor(total_num_images * train_rate)
    num_val = total_num_images - num_train
    print("Number of train images: "+str(num_train))
    print("Number of validation images: "+str(num_val))
    

    # Step 4: Create empty numpy arrays for both train and validation
    # Allocate space for X and Y
    X_train = np.empty((num_train, 1024, 1024, 32), np_precision)
    X_val = np.empty((num_val, 1024, 1024, 32), np_precision)
    Y_train = np.empty((num_train ), dtype=np_precision)
    Y_val = np.empty((num_val ), dtype=np_precision)




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

    # At this point, 
    # all_tiff_files_final is a flat, shuffled array containing all tiff files from every class 
    # all_class_final is a flat, shuffled array with matching class labels (0, 1) corresponding to each tiff file in all_tiff_files_final

    # Step 6: Finally, read through all tiff files and insert them into the training and validation arrays
    print("\nReading tiff files...")
    for i in range(0,len(file_index)):
        print("Reading tiff file: "+str(all_tiff_files_final[i])+" with class label "+str(all_class_final[i]))
        tiff_file = all_tiff_files_final[i]
        

        imgTiff = Image.open(tiff_file, mode="r")
        for j in range(32): # This will go through all 32 color channels
            imgTiff.seek(j)
            img = np.array(imgTiff, dtype=np_precision)*img_mult_amt
            if i < num_train:
                X_train[i,:,:,j] = img
            else:
                X_val[i-num_train,:,:,j] = img
        if i < num_train:
            Y_train[i] = all_class_final[i]
        else:
            Y_val[i-num_train] = all_class_final[i]
    
    print("Finished reading all tiff files")
    print("Size of arrays: ")
    print("X_train")
    print(X_train.shape)
    print("X_val")
    print(X_val.shape)
    print("Y_train")
    print(Y_train.shape)
    print("Y_val")
    print(Y_val.shape)
    print("Checking if there's any blank images...(this shouldn't return any errors)")
    any_blank_arrays = False
    for i in range(0, X_train.shape[0]):
        if np.count_nonzero(X_train[i,:,:,:]) == 0:
            any_blank_arrays = True 
            print("ERROR: non zero found in X_Train at index "+str(i))
    for i in range(0, X_val.shape[0]):
        if np.count_nonzero(X_val[i,:,:,:]) == 0:
            any_blank_arrays = True 
            print("ERROR: non zero found in X_val at index "+str(i))
    if not any_blank_arrays:
        print("Check was good, no blank images found!")
    #return np.swapaxes(X_train,1,3), Y_train, np.swapaxes(X_val,1,3), Y_val
    return X_train, Y_train, X_val, Y_val

def numTiffsInFolder(tiff_dir):
    num_tiffs = 0
    for file in os.listdir(tiff_dir):
        if ".tiff" in file:
            num_tiffs += 1
    return num_tiffs

#def main():
    # Just for testing
    #loadTiffFiles([["PBMC_DATA_JUL_2021/CSFpos_MCI/tiff", "PBMC_DATA_JUL_2021/CSFpos_AD/tiff"], ["PBMC_DATA_JUL_2021/CSFneg_healthy/tiff"]],  "/home/julianps/projects/def-stys/shared/", 0.75, np.float16,max_files=31)


#if __name__ == "__main__":
    #main()
