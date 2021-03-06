# Lf-project

This repository contains programs and related documents for our term project on "Restoring Low-Light Light Fields" done as part of the course, EE5176: Computational Photography at IIT Madras.

Team-Members:

+ [Snehal Singh Tomar](https://snehalstomar.github.io)
+ [Subhankar Chakraborty](https://github.com/Subhankar48)
+ [Aqil KH](https://github.com/AqilHussan)

### The Seven Broad steps required to be performed in order to prcess raw 4D light field data as mentioned in [1] are:

Note: Coordinates(i, j, k, l)[As mentioned in [1]] ~ (u, v, x, y)[As per theory taught in class]
1. Demosaicing(similar to bayer pattern) and Vignetting Correction(division by white image).
2. Aligning Sub-aperture(lenslet) image centres to integer lcations on the sensor grid(image centre--> brightest spot, for alignment--> Rotation, Scaling).
3. Slicing each of the lenslet images(While accounting for heaxagonal(specific to lytro camera) to Rectangular distortions) separately. This slicing happens in the outer(spatial) coordinates(x,y) and not in the inner(angular) coordinates(u, v).
4. Converting Hexagonally sampled data to a rectilinear grid by interpolating along x.
5. Correcting for rectangular(non-square) pixels by interpolating along u.
6. Masking off pixels that lie outside the hexagonal lenslet image
7. Final conversion (x,y,u,v) --> (u,v,x,y) [Interpreting the Light field as an array of images captured from different perspectives which form on a specific sized sub-grid of the Sensor plane.]  


### Details of our programs are as under:
1. Program to return decoded light field without saving it as .jpg/.png:
+ Function: decode_sans_saving.py
+ Function Execution: decode_main.py
+ Working Example: decode_sans_save.ipynb
+ Directory for storing raw LF data: Data/ 
2. Program to return decoded light field without performing demosaicing, contrast correction and AWB:
+ Function: Lftoolbox.py 
+ Function Execution: call_Lftoolbox.py
+ Working Example: testing_decoding_sans_saving.ipynb
+ Please ensure that 'Lftoolbox.py', 'call_Lftoolbox.py', the_raw_lightfield.lfr, and calib_data.tar are present in the PWD while executing 'call_Lftoolbox.py'



### References:
+ [1] D. G. Dansereau, O. Pizarro, and S. B. Williams, ???Decoding, calibration and rectification for lenselet-based plenoptic cameras,??? in Computer Vision and Pattern Recognition (CVPR), 2013, pp. 1027???1034.
+ [2] The [MATLAB light-field toolbox](https://in.mathworks.com/matlabcentral/fileexchange/75250-light-field-toolbox)
+ [3] Christopher Hahne and Amar Aggoun, "PlenoptiCam v1.0: A light-field imaging framework", arXiv, 2020. 
