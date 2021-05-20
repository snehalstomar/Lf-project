import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import plenopticam as pcam
import decode_sans_saving

loader = pcam.misc.DataDownloader()
loader.download_data(loader.host_eu_url, fp='Data')
loader.extract_archive(archive_fn='./data/illum_test_data.zip', fname_list='lfr')

lfp_path = 'Data/gradient_rose_close.lfr'
calibration_path = 'Data/caldata-B5144402350.tar'

#========Required output

decoded_views_all = decode_sans_saving.lf_decode_sans_save(lfp_path, calibration_path)

#========================

plt.imshow(decoded_views_all)
plt.show()