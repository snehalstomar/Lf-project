# Lf-project

This repository contains programs and related documents for our term project on "Restoring Low-Light Light Fields" done as part of the course, EE5176: Computational Photography at IIT Madras.

Team-Members:

+ Snehal Singh Tomar
+ Subhankar Chakraborty
+ Aqil KH

#The Seven Broad steps required to be performed in order to prcess raw 4D light field data as mentioned in [1] are:

Note: Coordinates(i, j, k, l)[As mentioned in [1]] ~ (u, v, x, y)[As per theory taught in class]
1. Demosaicing(similar to bayer pattern) and Vignetting Correction(division by white image).
2. Aligning Sub-aperutre(lenslet) image centres to integer lcations on the sensor grid(image centre--> brightest spot, for alignment--> Rotation, Scaling).
3. Slicing each of the lenslet images(While accounting for heaxagonal(specific to lytro camera) to Rectangular distortions) separately. This slicing happens in the outer(spatial) coordinates(x,y) and not in the inner(angular) coordinates(u, v).
4. Converting Hexagonally sampled data to a rectilinear grid by interpolating along x.
5. Correcting for rectangular(non-square) pixels by interpolating along u.
6. Masking off pixels that lie outside the hexagonal lenslet image
7. Final conversion (x,y,u,v) --> (u,v,x,y) [Interpreting the Light field as an array of images captured from different perspectives which form on a specific sized sub-grid of the Sensor plane.]  

#The functions from[2] which perform these tasks have been grouped as under:
+ Step 1:
+ Step 2:
+ Step 3:
+ Step 4:
+ Step 5:
+ Step 6:
+ Step 7:

#Our Python Translations for the same are listed as under:

| MATLAB Function   | Python Program |
| ----------------- | -------------- |
| A.m               | A.py           |
| B.m               | B.py           |



#References:
+ [1] D. G. Dansereau, O. Pizarro, and S. B. Williams, “Decoding, calibration and rectification for lenselet-based plenoptic cameras,” in Computer Vision and Pattern Recognition (CVPR), 2013, pp. 1027–1034.
+ [2] The [MATLAB light-field toolbox](https://in.mathworks.com/matlabcentral/fileexchange/75250-light-field-toolbox)