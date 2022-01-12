import numpy as np
import binascii
import struct
#from matplotlib import pyplot as plt
import time
import sys
import math
from struct import *

class Image_data():
    def __init__(self):
        self.metadata = []
        self.lwavelenghts = []
        self.endiancheck = -1
        self.version = -1
        self.numfloats = -1
        self.nx = -1
        self.ny = -1
        self.nz = -1
        self.total_pixels = -1
        self.nl = 32

        self.kernelintensity = 0
        self.checksum = 0
        self.kerneldim = 0
        self.img_data = []
        self.filename = ""

    # Setters
    def set_endiancheck(self, _endiancheck):
        self.endiancheck = _endiancheck

    def set_version(self, _version):
        self.version = _version

    def set_nx(self, _nx):
        self.nx = _nx

    def set_ny(self, _ny):
        self.ny = _ny

    def set_nz(self, _nz):
        self.nz = _nz

    def set_nl(self, _nl):
        self.nl = _nl

    def set_totalpixels(self, _total_pixels):
        self.total_pixels = _total_pixels

    def set_imgdata(self, _imgdata):
        self.img_data = _imgdata

    def set_filename(self,_filename):
        self.filename = _filename

    def set_numfloats(self,_numfloats):
        self.numfloats = _numfloats

    def set_metadata(self,_metadata):
        self.metadata = _metadata

    def set_lwavelengths(self,_lwavelengths):
        self.lwavelengths = _lwavelengths

    def set_kernelintensity(self,_kernelintensity):
        self.kernelintensity = _kernelintensity

    def set_kerneldim(self, _kerneldim):
        self.kerneldim = _kerneldim

    def set_checksum(self,_checksum):
        self.checksum = _checksum

    # Getters
    def get_endiancheck(self):
        return self.endiancheck

    def get_version(self):
        return self.version

    def get_nx(self):
        return self.nx

    def get_ny(self):
        return self.ny

    def get_nz(self):
        return self.nz

    def get_nl(self):
        return self.nl

    def get_totalpixels(self): # total pixels that are not black
        return self.total_pixels

    def get_imgdata(self):
        return self.img_data

    def get_filename(self):
        return self.filename

    def get_numfloats(self):
        return self.numfloats

    def get_metadata(self):
        return self.metadata

    def get_lwavelengths(self):
        return self.lwavelengths

    def get_kernelintensity(self):
        return self.kernelintensity

    def get_kerneldim(self):
        return self.kerneldim

    def get_checksum(self):
        return self.checksum


    # Other methods
    def shuffle_flatpixels(self):
        # only works for the flat image
        #self.img_data =
        #DOESN"T WORK
        np.random.shuffle(self.img_data)

    def print_all_data(self):
        print("Data dump:")
        print("File name: "+str(self.filename))
        print("endiancheck: "+str(self.endiancheck))
        print("version: "+str(self.version))
        print("nX: "+str(self.nx))
        print("nY: "+str(self.ny))
        print("nZ: "+str(self.nz))
        print("nL: "+str(self.nl))
        print("total pixels: "+str(self.total_pixels))
        print("number of floats: "+str(self.numfloats))
        print("metadata: "+str(self.metadata))
        print("wavelength info: "+str(self.lwavelengths))
        print("kernel intensity: "+str(self.kernelintensity))
        print("kernel dimmension: "+str(self.kerneldim))
        print("checksum: "+str(self.checksum))
        print("image shape: "+str(self.img_data.shape))


def byte_to_int(bytes):
    total = 0
    for byte in bytes:
        total = total * 256 +int(byte)
    return result


def import_binary(bin_path,filename, voxel_stuff=False, shuffle_voxels=False, only_read_header=False):
    # version 5
    # if voxel_stuff, then return flat np array of size
    np.set_printoptions(threshold=sys.maxsize)
    f = open(bin_path, "rb")

    # If true, then stop reading from binary file on next iteration
    should_stop_reading_data = False

    trace = 0
    # All parameters that need to keep track of
    endiancheck = 0
    version = 0
    kernel_intensity = 0
    checksum = 0
    nX = 0
    nY = 0
    nZ = 0
    nL = 0
    total_pixels = 0
    numfloats = 0
    kernel_dim = 0
    metadata = []
    l_wavelengths = []

    # Vars for reading pixel values from arrays
    # for storying a temp float
    float_temp_arr = []

    xCoord = 0
    yCoord = 0
    zCoord = 0

    lam_val = 0

    # Setting up image data array
    img_data = Image_data()

    # img_data_arr will hold all pixel data
    img_data_arr=np.empty((1,1))

    reading_meta_data = True
    all_data = f.read()
    all_data_bytes = bytearray(all_data)

    # Used for voxel stuffing
    pix_coor = 0

    index = 0
    while index < len(all_data_bytes):
        if should_stop_reading_data:
            break

        byte = all_data_bytes[index]

        if reading_meta_data:
            # Currently reading header info

            # endiancheck
            if trace < 4:
                endiancheck += (byte)*(256**((trace%4)))
            # version
            elif trace < 8:
                version+=(byte)*(256**((trace%4)))
            # nX
            elif trace < 12:
                nX+=(byte)*(256**((trace%4)))
            # nY
            elif trace < 16:
                nY+=(byte)*(256**((trace%4)))
            # nZ
            elif trace < 20:
                nZ+=(byte)*(256**((trace%4)))
            # nL
            elif trace < 24:
                nL+=(byte)*(256**((trace%4)))
            # total
            elif trace < 28:
                total_pixels+=(byte)*(256**((trace%4)))
            # kernel dim
            elif trace < 32:
                kernel_dim+=(byte)*(256**((trace%4)))
            # numfloats
            elif trace < 36:
                numfloats+=(byte)*(256**((trace%4)))
            # median integrated across NL bins kernel intensity of all pixels/kernels
            elif trace < 40:
                kernel_intensity += (byte)*(256**((trace%4)))
            # raw data checksum
            elif trace < 44:
                checksum += (byte)*(256**((trace%4)))

            elif trace >= (44+nL*4+numfloats*4):
                # At this point, we have finished reading meta data from header. Next, will start reading pixel values
                reading_meta_data = False
                if only_read_header:
                    should_stop_reading_data = True
                else:
                    if voxel_stuff:
                        # find size based on total pixels
                        img_data_arr = np.zeros((1, total_pixels, nL ),dtype=np.uint16)
                    else:
                        #img_data_arr = np.zeros((nX, nY, nL, nZ ),dtype=np.uint16)
                        img_data_arr = np.zeros((nX, nY, nL, nZ ),dtype=np.uint16)

                    index-=1
                    trace = -1
            else:
                # append meta data headers AND wavelength units
                float_temp_arr.append(byte)
                if trace >= (44+(numfloats+0)*4):
                    # wavelength
                    if len(float_temp_arr) == 4:
                        l_wavelengths.append(str(unpack("<f",bytes(float_temp_arr))[0]))
                        float_temp_arr = []
                else:
                    # header
                    if len(float_temp_arr) == 4:
                        metadata.append(str(unpack("<f",bytes(float_temp_arr))[0]))
                        float_temp_arr = []
            trace += 1

        else:
            # Currently reading pixel data

            # xCoord
            if trace < 2:
                xCoord+=(byte)*(256**((trace%2)))
            # yCoord
            elif trace < 4:
                yCoord+=(byte)*(256**((trace%2)))
            # zCoord
            elif trace < 6:
                zCoord+=(byte)*(256**((trace%2)))
            # pixel values
            else:
                # Have read x, y and z, now need to read lambda channel values, one by one
                lam_val+=(byte)*(256**((trace%2)))

                if (trace+1) % 2 == 0 and trace > 0:
                    # Fill up arrays with data that was read
                    if voxel_stuff:
                        # Stuff into flat array of size (1, num_pixels, nZ)
                        img_data_arr[0, pix_coor,  int((trace-7)/2)] = lam_val
                    else:
                        # Stuff pixels into (nX, nY, nL, nZ) array
                        for y in range(yCoord - math.floor(kernel_dim/2), yCoord + math.floor(kernel_dim/2)+1):
                            for x in range(xCoord - math.floor(kernel_dim/2), xCoord + math.floor(kernel_dim/2)+1):
                                #img_data_arr[x, y, int((trace-7)/2), zCoord] = lam_val
                                img_data_arr[x, y, int((trace-7)/2), zCoord] = lam_val

                    lam_val = 0
                if trace == 2*(3+nL)-1:
                    trace = -1 # -1
                    lam_val = 0

                    xCoord = 0
                    yCoord = 0
                    zCoord = 0

                    pix_coor += 1
            trace += 1
        index+=1

    # After reading all data, shufle in needed
    if shuffle_voxels and voxel_stuff:
        [np.random.shuffle(x) for x in img_data_arr]


    img_data.set_imgdata(img_data_arr)

    img_data.set_totalpixels(total_pixels)

    img_data.set_nz(nZ)
    img_data.set_ny(nY)
    img_data.set_nx(nX)
    img_data.set_nl(nL)

    img_data.set_version(version)

    img_data.set_filename(filename)
    img_data.set_numfloats(numfloats)
    img_data.set_metadata(metadata)
    img_data.set_endiancheck(endiancheck)
    img_data.set_lwavelengths(l_wavelengths)

    img_data.set_checksum(checksum)
    img_data.set_kernelintensity(kernel_intensity)
    img_data.set_kerneldim(kernel_dim)
    f.close()
    return img_data
