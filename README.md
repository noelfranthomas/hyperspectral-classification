# Hyperspectral Classification
Machine learning and statistical analysis for hyperspectral classification.

## Objective
Build ThunderSVM on CC network.

## Method(s)
1. Build environment off-site, transfer onto HPC network with singularity container.
2. Build adapter for ThunderSVM to use installed CUDA module version.

### July 14 
Following method 1:
1. Install linux (Ubuntu).
2. Create singularity image with appropriate CUDA.
3. Test ThunderSVM on local Ubuntu machine.

### July 13
1. Build ThunderSVM using Singularity
https://docs.sylabs.io/guides/3.5/user-guide/introduction.html
a. Create Dockerfile
b. Run container

### July 12
1. Build interface between ThunderSVM and CUDA 11.
    1. Write interface with C/C++
    2. Add ThunderSVM functionalities
    3. Publish package
