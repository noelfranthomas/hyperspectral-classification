import numpy as np
import binascii
import struct
#from matplotlib import pyplot as plt
import time
import sys
import os
import import_binary as imp
from PIL import Image
import math
#import tifffile



def main():
    max = 100000000
    bin_folder = "/Users/julianstys/Documents/ML/new/bin" # "/home/julianps/projects/def-stys/julianps/2021/test/binary_files/"
    tiff_folder = "/Users/julianstys/Documents/ML/new/tiff"  #"/home/julianps/projects/def-stys/julianps/2021/test/binary_files/out/"
    meta_file = "/Users/julianstys/Documents/ML/new/tiff/meta_data.txt"  # "/home/julianps/projects/def-stys/julianps/2021/test/binary_files/meta_data.txt"
    file_format = "/Users/julianstys/Documents/ML/new/v5_fileformat.txt"

    # If false, then read through binary and generate frames as normal
    # If true, use alternative method of stuffing new images so that no black space is present
    voxel_stuffing = False
    voxel_img_width = 20
    voxel_img_height = 20
    ignore_incomplete_file = False


    all_file_format_text = []

    try:
        file_format_lines = open(file_format).readlines()
    except:
        all_file_format_text = []
    else:
        for line in file_format_lines:
            all_file_format_text.append(line)

    print(all_file_format_text)

    if os.path.exists(meta_file):
        os.remove(meta_file)
    f = open(meta_file, "w+")
    i = 0


     # dump file format into meta data
    if len(all_file_format_text) > 0:
        f.write("----------Start of file format-----------\n")
        for line in all_file_format_text:
            f.write(line)
        f.write("-----------End of file format------------\n\n")



    for file in os.listdir(bin_folder):
        i += 1
        if i > max:
            break
        bin_path = bin_folder+"/"+file


        if "binary" in file:
            print("Reading:"+bin_path)
            img_data = imp.import_binary(bin_path,file,voxel_stuff=False)
            f.write("filename: "+str(img_data.get_filename())+"\n")
            f.write(" -version: "+str(img_data.get_version())+"\n")
            f.write(" -metadata: "+str(', '.join(img_data.get_metadata()))+"\n")
            f.write(" -wavelength info: "+str(', '.join(img_data.get_lwavelengths()))+"\n")
            f.write(" -endiancheck: "+str(img_data.get_endiancheck())+"\n")
            f.write(" -nx: "+str(img_data.get_nx())+"\n")
            f.write(" -ny: "+str(img_data.get_ny())+"\n")
            f.write(" -nz: "+str(img_data.get_nz())+"\n")
            f.write(" -nl: "+str(img_data.get_nl())+"\n")
            f.write(" -kernel dimension: "+str(img_data.get_kerneldim())+"\n")
            f.write(" -total pixels (all images): "+str(img_data.get_total())+"\n")
            f.write(" -median integrated kernel intensity of all pixels/kernels: "+str(img_data.get_kernelintensity())+"\n")
            f.write(" -raw data checksum: "+str(img_data.get_checksum())+"\n\n")

            #f.write(" -abeta142: "+str(img_data.get_abeta142())+"\n")
            #f.write(" -totaltau: "+str(img_data.get_totaltau())+"\n")
            #f.write(" -ptau: "+str(img_data.get_ptau())+"\n")
            #f.write(" -diagnosis: "+str(img_data.get_diagnosis())+"\n")
            #f.write(" -moca: "+str(img_data.get_moca())+"\n\n")

            #print(binary_arr.shape)
            #### Save as jpeg for viewing
            for i in range(0,img_data.get_imgdata().shape[3]):
                np_arr = img_data.get_imgdata()[:,:,:,i] #np.expand_dims(img_data.get_imgdata()[:,:,:,i],3)
                #print(np_arr.shape)
                np_arr=np.swapaxes(np_arr,0,2)
                #print(np_arr.shape)
                if np.count_nonzero(np_arr) == 0:
                    print("Warning: All zeros for: "+file+"_frame"+str(i)+".tiff")
                #else:

                tifffile.imsave(tiff_folder+"/"+file+"_frame"+str(i)+".tiff",np_arr)
                print("shape")
                print(np_arr.shape)
                saveImageAsJPEG(np_arr,i)
                exit()




    f.close()


def saveImagesAsJPEG(images):
    # Saves image as jpeg for viewing
    # images should be of the form (num_files, 32, 1024, 1024)
    # Approximates 32 channels as rgb for viewing
    print("save as jpeg")
    for i in range(0, images.shape[0]):
        print("saving image ")
        rgb_np = np.zeros((3, 1024, 1024), dtype=np.float32)
        rgb_np[2,:,:] = images[i,27,:,:]
        rgb_np[1,:,:] = images[i,14,:,:]
        rgb_np[0,:,:] = images[i,4,:,:]
        rgb_np = 255*rgb_np/np.amax(rgb_np)
        rgb_img = Image.fromarray(np.swapaxes(rgb_np.astype('uint8'),0,2), 'RGB')
        rgb_img.save(str(i)+".jpeg")



def linInterp(x, x0, x1, y0,y1):
    return y0 + (x - x0) * (y1-y0) / (x1-x0)


def saveImageAsJPEG(image, num):
    # Saves image as jpeg for viewing
    # images should be of the form (num_files, 32, 1024, 1024)
    # Approximates 32 channels as rgb for viewing
    i = 1
    print("save as jpeg")
    rgb_np = np.zeros((3, 1024, 1024), dtype=np.float32)
    for j in range(22,32):
        rgb_np[0,:,:] = rgb_np[0,:,:] +   image[j,:,:] #* math.sin( linInterp(j, 22, 32, 0, 3.1415)   )   # math.sin( linInterp(j, 25, 32, 0, 3.1415)   )*image[j,:,:]
        print("interp: ")
        print(j)
        print(linInterp(j, 22, 32, 0, 3.1415))
        print(math.sin( linInterp(j, 22, 32, 0, 3.1415)   ))
        print("--")
        #rgb_np[1,:,:] = rgb_np[1,:,:] + image[j,:,:]
    for j in range(11,22):
        rgb_np[1,:,:] = rgb_np[1,:,:] + image[j,:,:] #* math.sin( linInterp(j, 11, 22, 0, 3.1415)   )
    for j in range(0,11):
        rgb_np[2,:,:] = rgb_np[2,:,:] + image[j,:,:] #* math.sin( linInterp(j, 0, 11, 0, 3.1415)   )


    '''for j in range(0,32):
        rgb_np[0,:,:] = rgb_np[0,:,:]  + image[j,:,:]
        rgb_np[1,:,:] = rgb_np[1,:,:]  + image[j,:,:]
        rgb_np[2,:,:] = rgb_np[2,:,:]  + image[j,:,:]'''


    #rgb_np[2,:,:] = image[27,:,:] + image[28,:,:] + image[29,:,:] + image[30,:,:] + image[26,:,:] + image[25,:,:]
    #rgb_np[1,:,:] = image[14,:,:] + image[11,:,:] + image[12,:,:] + image[13,:,:] + image[15,:,:] + image[16,:,:]
    #rgb_np[0,:,:] = image[4,:,:] + image[2,:,:] + image[3,:,:] + image[5,:,:] + image[6,:,:] + image[7,:,:]
    print("MAX VAL")
    print(np.amax(rgb_np))
    rgb_np = rgb_np / np.amax(rgb_np)
    rgb_np = 255*rgb_np #/np.amax(rgb_np)
    #rgb_img = Image.fromarray(np.swapaxes(rgb_np.astype('uint8'),0,2), 'RGB')
    #rgb_img = Image.fromarray(np.swapaxes(rgb_np.astype('uint8'),0,2), 'RGB')
    rgb_np = np.swapaxes(rgb_np.astype('uint8'),0,2)
    #rgb_np = np.swapaxes(rgb_np.astype('uint8'),0,1)
    rgb_img = Image.fromarray(np.swapaxes(rgb_np.astype('uint8'),0,1), 'RGB')

    rgb_img.save(str(i)+"_"+str(num)+".png")


'''
for k = 0 to nL-1   ' lambda ch ctr
    dim p as Ptr = frdPtrs(k)   ' Ptr to UInt16 image matrix @ kth wavelength
    for yKernelCtr as integer = yCoord-kernelDimDIV2 to yCoord+kernelDimDIV2
      for xKernelCtr as integer = xCoord-kernelDimDIV2 to xCoord+kernelDimDIV2
        p.UInt16(2*(xKernelCtr + nX*yKernelCtr)) = pixVals(k)
      next
    next
  next
'''


'''
kernaldim = round(kernaldim / 2)
for each lamdba channel:
    get pointer to image matrix p
    for y in range( yCenterKernel - kernaldim, yCenterKernel + kernaldim )
        for x in range( xCenterKernel-kernaldim, xCenterKernel+ kernaldim  )


'''

if __name__ == "__main__":
    main()
