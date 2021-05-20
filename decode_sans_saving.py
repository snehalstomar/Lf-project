import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
	import plenopticam as pcam
except ImportError:
	print('run-> $python3 -m pip install plenopticam>=0.7.0')

def lf_decode_sans_save(lfp_path, cal_path, full_sai = True, central_view_extract_dim = 3):
	#Configuration
	cfg = pcam.cfg.PlenopticamConfig()
	cfg.default_values()
	cfg.params[cfg.lfp_path] = lfp_path
	cfg.params[cfg.cal_path] = cal_path
	cfg.params[cfg.opt_cali] = True
	cfg.params[cfg.ptc_leng] = 13
	cfg.params[cfg.cal_meth] = pcam.cfg.constants.CALI_METH[3]
	sta = pcam.misc.PlenopticamStatus()

	#reading raw LF
	reader = pcam.lfp_reader.LfpReader(cfg, sta)
	reader.main()
	lfp_img = reader.lfp_img

	#finding appropriate white image in the calibration data
	cal_finder = pcam.lfp_calibrator.CaliFinder(cfg, sta)
	ret = cal_finder.main()
	wht_img = cal_finder.wht_bay

	#Micro Image Calibration
	cal_obj = pcam.lfp_calibrator.LfpCalibrator(wht_img, cfg, sta)
	ret = cal_obj.main()
	cfg = cal_obj.cfg

	#Micro Image Alignment
	ret = cfg.load_cal_data()
	aligner = pcam.lfp_aligner.LfpAligner(lfp_img, cfg, sta, wht_img)
	ret = aligner.main()
	lfp_img_align = aligner.lfp_img

	#Extracting Sub Aperture Images
	extractor = pcam.lfp_extractor.LfpExtractor(lfp_img_align, cfg, sta)
	ret = extractor.main()
	vp_img_arr = extractor.vp_img_arr

	view_obj = pcam.lfp_extractor.LfpViewpoints(vp_img_arr=vp_img_arr)
	vp_view = view_obj.central_view

	#Extracting all sub aperture views and displaying them
	view_obj = pcam.lfp_extractor.LfpViewpoints(vp_img_arr=vp_img_arr)
	vp_stack = view_obj.views_stacked_img
	vp_stack_out = vp_stack/vp_stack.max()

	if full_sai == True:
		return vp_stack_out


	





