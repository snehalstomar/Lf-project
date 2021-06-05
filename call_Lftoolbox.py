import plenopticam as pcam
import numpy as np
import Lftoolbox


lfp_path = '/content/drive/MyDrive/EE5176/IMG_1108.lfr'
calibration_path = '/content/drive/MyDrive/EE5176/caldata-B5152300590.tar'
#function call. desired output = decoded_views_all
decoded_views_all = Lftoolbox.lf_decode_sans_save(lfp_path, calibration_path)
#visualization of results
plt.imshow(decoded_views_all,cmap='gray')
plt.show()