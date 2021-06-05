import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import plenopticam as pcam



#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "info@christopherhahne.de"
__license__ = """
Copyright (c) 2019 Christopher Hahne <info@christopherhahne.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

from plenopticam.cfg import PlenopticamConfig
from plenopticam import misc

import numpy as np
import os


class LfpMicroLenses(object):

    def __init__(self, *args, **kwargs):

        # variables
        self._lfp_img = kwargs['lfp_img'] if 'lfp_img' in kwargs else None
        self._wht_img = kwargs['wht_img'] if 'wht_img' in kwargs else None
        self._lfp_img_align = kwargs['lfp_img_align'] if 'lfp_img_align' in kwargs else None
        self.cfg = kwargs['cfg'] if 'cfg' in kwargs else PlenopticamConfig()
        self.sta = kwargs['sta'] if 'sta' in kwargs else misc.PlenopticamStatus()
        self._M = 0
        self._C = 0
        self._flip = kwargs['flip'] if 'flip' in kwargs else False

        # convert to float
        self._lfp_img = self._lfp_img.astype('float64') if self._lfp_img is not None else None
        self._wht_img = self._wht_img.astype('float64') if self._wht_img is not None else None

        if self.cfg.calibs:
            # micro lens array variables
            self._CENTROIDS = np.asarray(self.cfg.calibs[self.cfg.mic_list])
            self._LENS_Y_MAX = int(max(self._CENTROIDS[:, 2])+1)    # +1 to account for index 0
            self._LENS_X_MAX = int(max(self._CENTROIDS[:, 3])+1)    # +1 to account for index 0

        # get pitch from aligned light field
        self._M = self.lfp_align_pitch() if hasattr(self, '_lfp_img') else self._M

        # get mean pitch from centroids
        mean_pitch = self.centroid_avg_pitch(self._CENTROIDS) if hasattr(self, '_CENTROIDS') else self._M

        # evaluate mean pitch size and user pitch size
        self._Mn = self.safe_pitch_eval(mean_pitch=mean_pitch, user_pitch=int(self.cfg.params[self.cfg.ptc_leng]))

        # check if chosen micro image size too large
        if 0 < self._M < self._Mn:
            # remove existing pickle file
            fp = os.path.join(self.cfg.exp_path, 'lfp_img_align.pkl')
            os.remove(fp)
            # status update
            self.sta.status_msg('Angular resolution mismatch in previous alignment. Redo process')
            self.sta.error = True
        # check if micro image size in valid range
        elif self._M >= self._Mn > 0:
            self.cfg.params[self.cfg.ptc_leng] = self._Mn
        # check if micro image size not set
        elif self._M == 0:
            self._M = self._Mn
            self.cfg.params[self.cfg.ptc_leng] = self._Mn

        self._C = self._M // 2

        try:
            self._DIMS = self._lfp_img.shape if len(self._lfp_img.shape) == 3 else self._lfp_img.shape + (1,)
        except (TypeError, AttributeError):
            pass
        except IndexError:
            self.sta.status_msg('Incompatible image dimensions: Please either use KxLx3 or KxLx1 array dimensions')
            self.sta.error = True

    def proc_lens_iter(self, fun, **kwargs):
        """ process light-field based on provided function handle and argument data """

        # status message handling
        msg = kwargs['msg'] if 'msg' in kwargs else 'Light-field alignment process'
        usr_prnt = kwargs['prnt'] if 'prnt' in kwargs else True
        if usr_prnt:
            self.sta.status_msg(msg, self.cfg.params[self.cfg.opt_prnt])

        args = [kwargs[key] for key in kwargs.keys() if key not in ('cfg', 'sta', 'msg', 'prnt')]

        try:
            # iterate over each MIC
            for ly in range(self._LENS_Y_MAX):
                for lx in range(self._LENS_X_MAX):

                    # perform provided function
                    fun(ly, lx, *args)

                # print progress status
                self.sta.progress((ly + 1) / self._LENS_Y_MAX * 100, self.cfg.params[self.cfg.opt_prnt])

                # check interrupt status
                if self.sta.interrupt:
                    return False

        except Exception as e:
            raise e

        return True

    def get_coords_by_idx(self, ly: int, lx: int) -> (float, float):
        """ yields micro image center in 2-D image coordinates """

        # filter mic by provided indices
        mic = self._CENTROIDS[(self._CENTROIDS[:, 2] == ly) & (self._CENTROIDS[:, 3] == lx), [0, 1]]

        return mic[0], mic[1]

    def safe_pitch_eval(self, mean_pitch: float, user_pitch: int) -> int:
        """ provide odd pitch size that is safe to use """

        # ensure patch size and mean patch size are odd
        mean_pitch += np.mod(mean_pitch, 2) - 1
        user_pitch += np.mod(user_pitch, 2) - 1
        safe_pitch = 3

        # comparison of patch size and mean size
        if safe_pitch <= user_pitch <= mean_pitch+2:  # allow user pitch to be slightly bigger than estimate
            safe_pitch = user_pitch
        elif user_pitch > mean_pitch:
            safe_pitch = mean_pitch
            msg_str = 'Patch size ({0} px) is larger than micro image size and reduced to {1} pixels.'
            self.sta.status_msg(msg_str.format(user_pitch, mean_pitch), self.cfg.params[self.cfg.opt_prnt])
        elif user_pitch < safe_pitch < mean_pitch:
            safe_pitch = mean_pitch
            msg_str = 'Patch size ({0} px) is too small and increased to {1} pixels.'
            self.sta.status_msg(msg_str.format(user_pitch, mean_pitch), self.cfg.params[self.cfg.opt_prnt])
        elif user_pitch < safe_pitch and mean_pitch < safe_pitch:
            self.sta.status_msg('Micro image dimensions are too small for light field computation.', True)
            self.sta.interrupt = True

        return int(safe_pitch)

    @staticmethod
    def centroid_avg_pitch(centroids: (list, np.ndarray)) -> int:
        """ estimate micro image pitch only from centroids """

        # convert to numpy array
        centroids = np.asarray(centroids)

        # estimate maximum patch size
        central_row_idx = int(max(centroids[:, 3])/2)
        mean_pitch = int(np.ceil(np.mean(np.diff(centroids[centroids[:, 3] == central_row_idx, 0]))))

        # ensure mean patch size is odd
        mean_pitch += np.mod(mean_pitch, 2)-1

        return int(mean_pitch)

    def centroid_align_pitch(self) -> int:
        """ obtain micro image pitch of aligned light-field from number of centroids """

        # estimate patch size
        lens_max_y = self._CENTROIDS[:][2].max() + 1     # +1 to account for index 0
        lens_max_x = self._CENTROIDS[:][3].max() + 1     # +1 to account for index 0
        pitch_estimate_y = self._lfp_img.shape[0]/lens_max_y
        pitch_estimate_x = self._lfp_img.shape[1]/lens_max_x

        if pitch_estimate_y-int(pitch_estimate_y) != 0 or pitch_estimate_x-int(pitch_estimate_x) != 0:
            msg = 'Micro image patch size error. Remove output folder or select re-calibration in settings.'
            self.sta.status_msg(msg=msg, opt=self.cfg.params[self.cfg.opt_prnt])
            self.sta.error = True

        return int(pitch_estimate_y)

    def lfp_align_pitch(self) -> int:
        """ estimate pitch size from aligned light-field (when centroids not available) """

        # initialize output variable (return zero if light field not present)
        res = 0
        if self._lfp_img_align is None:
            return res

        # use vertical dimension only (as horizontal may differ from hexagonal stretching)
        if hasattr(self, '_LENS_Y_MAX'):
            res = int(self._lfp_img_align.shape[0] / self._LENS_Y_MAX)
        else:
            # iterate through potential (uneven) micro image size candidates
            for d in np.arange(3, 51, 2):
                # take pitch where remainder of ratio between aligned image dimensions and candidate size is zero
                if (self._lfp_img_align.shape[0] / d) % 1 == 0 and (self._lfp_img_align.shape[1] / d) % 1 == 0:
                    res = int(d)
                    break

        return res

    @staticmethod
    def get_hex_direction(centroids: np.ndarray) -> bool:
        """ check if lower neighbor of upper left micro image center is shifted to left or right in hex grid

        :param centroids: phased array data
        :return: True if shifted to right
        """

        # get upper left MIC
        first_mic = centroids[(centroids[:, 2] == 0) & (centroids[:, 3] == 0), [0, 1]]

        # retrieve horizontal micro image shift (to determine search range borders)
        central_row_idx = int(centroids[:, 3].max()/2)
        mean_pitch = np.mean(np.diff(centroids[centroids[:, 3] == central_row_idx, 0]))

        # try to find MIC in lower left range (considering hexagonal order)
        found_mic = centroids[(centroids[:, 0] > first_mic[0]+mean_pitch/2) &
                              (centroids[:, 0] < first_mic[0]+3*mean_pitch/2) &
                              (centroids[:, 1] < first_mic[1]) &
                              (centroids[:, 1] > first_mic[1]-3*mean_pitch/4)].ravel()

        # true if MIC of next row lies on the right (false otherwise)
        hex_odd = True if found_mic.size == 0 else False

        return hex_odd

    @property
    def lfp_img(self):
        return self._lfp_img.copy() if self._lfp_img is not None else False

    @property
    def lfp_img_align(self):
        return self._lfp_img_align.copy() if self._lfp_img_align is not None else None

# local imports
from plenopticam import misc
from plenopticam.lfp_aligner.lfp_resampler import LfpResampler
from plenopticam.lfp_aligner.lfp_rotator import LfpRotator
from plenopticam.lfp_aligner.cfa_outliers import CfaOutliers
from plenopticam.lfp_aligner.cfa_processor import CfaProcessor
from plenopticam.lfp_aligner.lfp_devignetter import LfpDevignetter


class LfpAlignerModified(object):

    def __init__(self, lfp_img, cfg=None, sta=None, wht_img=None):

        # input variables
        self.cfg = cfg
        self.sta = sta if sta is not None else misc.PlenopticamStatus()
        self._lfp_img = lfp_img.astype('float') if lfp_img is not None else None
        self._wht_img = wht_img.astype('float') if wht_img is not None else None

    def main(self):

        if self.cfg.lfpimg:
            # hot pixel correction
            obj = CfaOutliers(bay_img=self._lfp_img, cfg=self.cfg, sta=self.sta)
            obj.rectify_candidates_bayer(n=9, sig_lev=2.5)
            self._lfp_img = obj.bay_img
            del obj

        if self.cfg.params[self.cfg.opt_vign] and self._wht_img is not None:
            # apply de-vignetting
            obj = LfpDevignetter(lfp_img=self._lfp_img, wht_img=self._wht_img, cfg=self.cfg, sta=self.sta)
            obj.main()
            self._lfp_img = obj.lfp_img
            self._wht_img = obj.wht_img
            del obj

      #  if self.cfg.lfpimg and len(self._lfp_img.shape) == 2:
            # perform color filter array management and obtain rgb image
       #     cfa_obj = CfaProcessor(bay_img=self._lfp_img, wht_img=self._wht_img, cfg=self.cfg, sta=self.sta)
        #    cfa_obj.main()
         #   self._lfp_img = cfa_obj.rgb_img
          #  del cfa_obj

        if self.cfg.params[self.cfg.opt_rota] and self._lfp_img is not None:
            # de-rotate centroids
            obj = LfpRotator(self._lfp_img, self.cfg.calibs[self.cfg.mic_list], rad=None, cfg=self.cfg, sta=self.sta)
            obj.main()
            self._lfp_img, self.cfg.calibs[self.cfg.mic_list] = obj.lfp_img, obj.centroids
            del obj

        # interpolate each micro image with its MIC as the center with consistent micro image size
        obj = LfpResamplerModified(lfp_img=self._lfp_img, cfg=self.cfg, sta=self.sta, method='linear')
        obj.main()
        self._lfp_img = obj.lfp_out
        del obj

        return True

    @property
    def lfp_img(self):
        return self._lfp_img.copy()



# local imports
from plenopticam import misc
from plenopticam.misc.type_checks import rint


# external libs
import numpy as np
import os
import pickle
import functools
from scipy.interpolate import interp2d, RectBivariateSpline


class LfpResamplerModified(LfpMicroLenses):

    def __init__(self, *args, **kwargs):
        super(LfpResamplerModified, self).__init__(*args, **kwargs)



        # interpolation method initialization
        method = kwargs['method'] if 'method' in kwargs else None
        method = method if method in ['nearest', 'linear', 'cubic', 'quintic'] else None
        method = 'cubic' if method == 'quintic' and self._M < 5 else method
        interp2d_method = functools.partial(interp2d, kind=method) if method is not None else interp2d

        if method is None:
            self._interpol_method = RectBivariateSpline
        elif method == 'nearest':
            self._interpol_method = self._nearest
        else:
            self._interpol_method = interp2d_method

        # output variable
        self._lfp_out = np.zeros(self._lfp_img.shape) if self._lfp_img is not None else None

    def main(self):
        """ cropping micro images to square shape while interpolating around their detected center (MIC) """

        # check interrupt status
        if self.sta.interrupt:
            return False

        # print status
        self.sta.status_msg('Light-field alignment', self.cfg.params[self.cfg.opt_prnt])

        # start resampling process (taking micro lens arrangement into account)
        if self.cfg.calibs[self.cfg.pat_type] == 'rec':
            self.resample_rec()
        elif self.cfg.calibs[self.cfg.pat_type] == 'hex':
            self.resample_hex()

        # save aligned image to hard drive
        self._write_lfp_align()

        return True

    def _write_lfp_align(self):

        # print status
        self.sta.status_msg('Save aligned light-field', self.cfg.params[self.cfg.opt_prnt])
        self.sta.progress(None, self.cfg.params[self.cfg.opt_prnt])

        # convert to 16bit unsigned integer
        self._lfp_out = misc.Normalizer(self._lfp_out).uint16_norm()

        # create output data folder
        misc.mkdir_p(self.cfg.exp_path, self.cfg.params[self.cfg.opt_prnt])

        # write aligned light field as pickle file to avoid re-calculation
        with open(os.path.join(self.cfg.exp_path, 'lfp_img_align.pkl'), 'wb') as f:
            pickle.dump(self._lfp_out, f)

        if self.cfg.params[self.cfg.opt_dbug]:
            misc.save_img_file(self._lfp_out, os.path.join(self.cfg.exp_path, 'lfp_img_align.tiff'))

        self.sta.progress(100, self.cfg.params[self.cfg.opt_prnt])

    def _patch_align(self, window, mic):

        # initialize patch
        patch = np.zeros(window.shape)
        

        # verify patch shapes as wrong shapes cause crashes
        if window.shape[0] == self._M+2 and window.shape[1] == self._M+2:
            # iterate through color channels
           # for p in range(self._DIMS[2]):

                fun = self._interpol_method(range(window.shape[1]), range(window.shape[0]), window[:, :])

                patch[:, :] = fun(np.arange(window.shape[1])+mic[1]-rint(mic[1]),
                                     np.arange(window.shape[0])+mic[0]-rint(mic[0]))
        else:
            self.sta.status_msg('Warning: chosen micro image size exceeds light-field borders')
            return np.zeros((self._M+2,)*2)

        # flip patch to compensate for micro lens rotation
        patch = np.flip(patch, axis=(0, 1)) if self._flip else patch

        return patch

    def _nearest(self, range0, range1, window):

        def shift_win(shifted_range0, shifted_range1):
            range0 = np.round(shifted_range0).astype('int')
            range1 = np.round(shifted_range1).astype('int')
            return window[range0[0]:range0[-1]+1, range1[0]:range0[-1]+1]

        return shift_win

    def resample_rec(self):

        # initialize variables required for micro image resampling process
        self._lfp_out = np.zeros([self._LENS_Y_MAX * self._M, self._LENS_X_MAX * self._M, self._DIMS[2]])

        # iterate over each MIC
        for ly in range(self._LENS_Y_MAX):
            for lx in range(self._LENS_X_MAX):

                # find MIC by indices
                mic = self.get_coords_by_idx(ly=ly, lx=lx)

                # interpolate each micro image with its MIC as the center with consistent micro image size
                window = self._lfp_img[rint(mic[0])-self._C-1:rint(mic[0])+self._C+2,
                                       rint(mic[1])-self._C-1:rint(mic[1])+self._C+2]
                self._lfp_out[ly*self._M:(ly+1)*self._M, lx*self._M:(lx+1)*self._M] = \
                    self._patch_align(window, mic)[1:-1, 1:-1]

            # check interrupt status
            if self.sta.interrupt:
                return False

            # print progress status for on console
            self.sta.progress((ly + 1) / self._LENS_Y_MAX * 100, self.cfg.params[self.cfg.opt_prnt])

        return True

    def resample_hex(self):

        # initialize variables required for micro image resampling process
        patch_stack = np.zeros([self._LENS_X_MAX, self._M, self._M])
        hex_stretch = int(np.round(2 * self._LENS_X_MAX / np.sqrt(3)))
        interp_stack = np.zeros([hex_stretch, self._M, self._M])
        self._lfp_out = np.zeros([self._LENS_Y_MAX * self._M, hex_stretch * self._M])

        # check if lower neighbor of upper left MIC is shifted to left or right
        hex_odd = self.get_hex_direction(self._CENTROIDS)

        # iterate over each MIC
        for ly in range(self._LENS_Y_MAX):
            for lx in range(self._LENS_X_MAX):

                # find MIC by indices
                mic = self.get_coords_by_idx(ly=ly, lx=lx)

                # interpolate each micro image with its MIC as the center and consistent micro image size
                window = self._lfp_img[rint(mic[0])-self._C-1:rint(mic[0])+self._C+2,
                                       rint(mic[1])-self._C-1:rint(mic[1])+self._C+2]
                patch_stack[lx, :] = self._patch_align(window, mic)[1:-1, 1:-1]

            # image stretch interpolation in x-direction to compensate for hex-alignment
            for y in range(self._M):
                for x in range(self._M):
                    #for p in range(self._DIMS[2]):
                        # stack of micro images elongated in x-direction
                        interp_coords = np.linspace(0, self._LENS_X_MAX, int(np.round(self._LENS_X_MAX*2/np.sqrt(3))))+\
                                        .5*np.mod(ly+hex_odd, 2)
                        interp_stack[:, y, x] = np.interp(interp_coords, range(self._LENS_X_MAX), patch_stack[:, y, x])

            self._lfp_out[ly*self._M:(ly+1)*self._M, :] = \
                np.concatenate(interp_stack, axis=1).reshape((self._M, hex_stretch*self._M))

            # check interrupt status
            if self.sta.interrupt:
                return False

            # print progress status
            self.sta.progress((ly+1) / self._LENS_Y_MAX * 100, self.cfg.params[self.cfg.opt_prnt])

    @property
    def lfp_out(self):
        return self._lfp_out.copy()

import numpy as np


from plenopticam.misc import PlenopticamError


from plenopticam.cfg import PlenopticamConfig
from plenopticam.misc import PlenopticamStatus
from plenopticam.misc.circle_drawer import bresenham_circle

import numpy as np


class LfpViewpoints(object):

    def __init__(self, *args, **kwargs):

        self._vp_img_arr = kwargs['vp_img_arr'] if 'vp_img_arr' in kwargs else None
        self._vp_img_arr = self.vp_img_arr.astype('float64') if self.vp_img_arr is not None else None
        self.cfg = kwargs['cfg'] if 'cfg' in kwargs else PlenopticamConfig()
        self.sta = kwargs['sta'] if 'sta' in kwargs else PlenopticamStatus()
        self._M = self.cfg.params[self.cfg.ptc_leng]
        self._C = self._M // 2

        try:
            self._DIMS = self._vp_img_arr.shape if len(self._vp_img_arr.shape) == 3 else self._vp_img_arr.shape + (1,)
        except (TypeError, AttributeError):
            pass
        except IndexError:
            self.sta.status_msg('Incompatible image dimensions: Please either use KxLx3 or KxLx1 array dimensions')
            self.sta.error = True

    @property
    def vp_img_arr(self):
        return self._vp_img_arr

    @vp_img_arr.setter
    def vp_img_arr(self, vp_img_arr):
        self._vp_img_arr = vp_img_arr

    @property
    def central_view(self):
        return self._vp_img_arr[self._C, self._C, ...].copy() if self._vp_img_arr is not None else None

    @staticmethod
    def remove_proc_keys(kwargs, data_type=None):

        data_type = dict if not data_type else data_type
        keys_to_remove = ('cfg', 'sta', 'msg', 'iter_num', 'iter_tot')

        if data_type == dict:
            output = dict((key, kwargs[key]) for key in kwargs if key not in keys_to_remove)
        elif data_type == list:
            output = list(kwargs[key] for key in kwargs.keys() if key not in keys_to_remove)
        else:
            output = None

        return output

    def proc_vp_arr(self, fun, **kwargs):
        """ process viewpoint images based on provided function handle and argument data """

        # percentage indices for tasks having sub-processes
        iter_num = kwargs['iter_num'] if 'iter_num' in kwargs else 0
        iter_tot = kwargs['iter_tot'] if 'iter_tot' in kwargs else 1

        # status message handling
        self.sta.progress(None, self.cfg.params[self.cfg.opt_prnt])
        if iter_num == 0:
            msg = kwargs['msg'] if 'msg' in kwargs else 'Viewpoint process'
            self.sta.status_msg(msg, self.cfg.params[self.cfg.opt_prnt])

        args = self.remove_proc_keys(kwargs, data_type=list)

        # light-field shape handling
        if len(self.vp_img_arr.shape) != 5:
            raise NotImplementedError
        new_shape = fun(self._vp_img_arr[0, 0, ...].copy(), *args).shape
        new_array = np.zeros(self._vp_img_arr.shape[:2] + new_shape)

        for j in range(self._vp_img_arr.shape[0]):
            for i in range(self._vp_img_arr.shape[1]):

                # check interrupt status
                if self.sta.interrupt:
                    return False

                res = fun(self._vp_img_arr[j, i, ...], *args)

                if res.shape == self._vp_img_arr.shape:
                    self._vp_img_arr[j, i, ...] = res
                else:
                    new_array[j, i, ...] = res

                # progress update
                percent = (j*self._vp_img_arr.shape[1]+i+1)/np.dot(*self._vp_img_arr.shape[:2])
                percent = percent / iter_tot + iter_num / iter_tot
                self.sta.progress(percent*100, self.cfg.params[self.cfg.opt_prnt])

        if new_array.sum() != 0:
            self._vp_img_arr = new_array

        return True

    @staticmethod
    def get_move_coords(arr_dims: (int, int) = (None, None), pattern: str = None, r: int = None) -> list:
        """ compute view coordinates that are used for loop iterations """

        # parameter initialization
        pattern = 'circle' if pattern is None else pattern
        r = r if r is not None else min(arr_dims)//2
        mask = [[0] * arr_dims[1] for _ in range(arr_dims[0])]

        if pattern == 'square':
            mask[0, :] = 1
            mask[:, 0] = 1
            mask[-1, :] = 1
            mask[:, -1] = 1
        if pattern == 'circle':
            mask = bresenham_circle(arr_dims, r=r)

        # extract coordinates from mask
        coords_table = [(y, x) for y in range(len(mask)) for x in range(len(mask)) if mask[y][x]]

        # sort coordinates in angular order
        coords_table.sort(key=lambda coords: np.arctan2(coords[0] - arr_dims[0]//2, coords[1] - arr_dims[1]//2))

        return coords_table

    def reorder_vp_arr(self, pattern=None, lf_radius=None):

        # parameter initialization
        pattern = 'circle' if pattern is None else pattern
        move_coords = self.get_move_coords(arr_dims=self.vp_img_arr.shape[:2], pattern=pattern, r=lf_radius)

        vp_img_set = []
        for coords in move_coords:
            vp_img_set.append(self.vp_img_arr[coords[0], coords[1], ...])

        return vp_img_set

    def proc_ax_propagate_1d(self, fun, idx=None, axis=None, **kwargs):
        """ apply provided function along axis direction """

        # status message handling
        if 'msg' in kwargs:
            self.sta.status_msg(kwargs['msg'], self.cfg.params[self.cfg.opt_prnt])

        axis = 0 if axis is None else axis
        j = 0 if idx is None else idx
        m, n = (0, 1) if axis == 0 else (1, 0)
        p, q = (1, -1) if axis == 0 else (-1, 1)

        for i in range(self._C):

            # swap axes indices
            j, i = (i, j) if axis == 1 else (j, i)

            ref_pos = self.vp_img_arr[self._C + j, self._C + i, ...]
            ref_neg = self.vp_img_arr[self._C + j * p, self._C + i * q, ...]

            self._vp_img_arr[self._C + j + m, self._C + i + n, ...] = \
                fun(self.vp_img_arr[self._C + j + m, self._C + i + n, ...], ref_pos, **kwargs)
            self._vp_img_arr[self._C + (j + m) * p, self._C + (i + n) * q, ...] = \
                fun(self.vp_img_arr[self._C + (j + m) * p, self._C + (i + n) * q, ...], ref_neg, **kwargs)

            # swap axes indices
            j, i = (i, j) if axis == 1 else (j, i)

            # check interrupt status
            if self.sta.interrupt:
                return False

        return True

    def proc_ax_propagate_2d(self, fun, **kwargs):
        """ apply provided function along axes """

        # percentage indices for tasks having sub-processes
        iter_num = kwargs['iter_num'] if 'iter_num' in kwargs else 0
        iter_tot = kwargs['iter_tot'] if 'iter_tot' in kwargs else 1

        # status message handling
        if iter_num == 0:
            msg = kwargs['msg'] if 'msg' in kwargs else 'Viewpoint process'
            self.sta.status_msg(msg, self.cfg.params[self.cfg.opt_prnt])

        kwargs = self.remove_proc_keys(kwargs, data_type=dict)

        self.proc_ax_propagate_1d(fun, idx=0, axis=0, **kwargs)

        for j in range(-self._C, self._C + 1):

            # apply histogram matching along entire column
            self.proc_ax_propagate_1d(fun, idx=j, axis=1, **kwargs)

            # progress update
            percent = (j + self._C + 1) / self._vp_img_arr.shape[0]
            percent = percent / iter_tot + iter_num / iter_tot
            self.sta.progress(percent*100, self.cfg.params[self.cfg.opt_prnt])

            # check interrupt status
            if self.sta.interrupt:
                return False

        return True

    @property
    def views_stacked_img(self):
        """ concatenation of all sub-aperture images for single image representation """
        return np.moveaxis(np.concatenate(np.moveaxis(np.concatenate(np.moveaxis(self.vp_img_arr, 1, 2)), 0, 2)), 0, 1)

    def circular_view_aperture(self, offset=None, ellipse=None):

        # initialize variables
        offset = offset if offset is not None else 0
        ratio = self.vp_img_arr.shape[3]/self.vp_img_arr.shape[2] if ellipse else 1
        r = self._M // 2
        mask = np.zeros([2*r+1, 2*r+1])

        # determine mask for affected views
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                if int(np.round(np.sqrt(x ** 2 + y ** 2 * ratio))) > r + offset:
                    mask[r + y][r + x] = 1

        # extract coordinates from mask
        coords_table = [(y, x) for y in range(len(mask)) for x in range(len(mask)) if mask[y][x]]

        # zero-out selected views
        for coords in coords_table:
            self.vp_img_arr[coords[0], coords[1], ...] = np.zeros(self.vp_img_arr.shape[2:])

        return True




class LfpRearrangerModified(LfpViewpoints):

    def __init__(self, lfp_img_align=None, *args, **kwargs):
        super(LfpRearrangerModified, self).__init__(*args, **kwargs)

        self._lfp_img_align = lfp_img_align if lfp_img_align is not None else None
        self._dtype = self._lfp_img_align.dtype if self._lfp_img_align is not None else self._vp_img_arr.dtype

    def _init_vp_img_arr(self):
        """ initialize viewpoint output image array """

        if len(self._lfp_img_align.shape) == 3:
            m, n, p = self._lfp_img_align.shape
        elif len(self._lfp_img_align.shape) == 2:
            m, n, p = self._lfp_img_align.shape[:2] + (1,)
        else:
            raise PlenopticamError('Dimensions %s of provided light-field not supported', self._lfp_img_align.shape,
                                   cfg=self.cfg, sta=self.sta)

        self._vp_img_arr = np.zeros([int(self._M), int(self._M), int(m/self._M), int(n/self._M)], dtype=self._dtype)

    def _init_lfp_img_align(self):
        """ initialize micro image output image array """

        if len(self._vp_img_arr.shape) == 4:
            m, n, p = self._vp_img_arr.shape[2:]
        elif len(self._vp_img_arr.shape) == 3:
            m, n, p = self._vp_img_arr.shape[2:] + (1,)
        else:
            raise PlenopticamError('Dimensions %s of provided light-field not supported', self._vp_img_arr.shape,
                                   cfg=self.cfg, sta=self.sta)

        m *= self._vp_img_arr.shape[0]
        n *= self._vp_img_arr.shape[1]

        # create empty array
        self._lfp_img_align = np.zeros([m, n], dtype=self._dtype)

        # update angular resolution parameter
        self._M = self._vp_img_arr.shape[0] if self._vp_img_arr.shape[0] == self._vp_img_arr.shape[1] else float('inf')

    def main(self):

        # check interrupt status
        if self.sta.interrupt:
            return False

        # rearrange light-field to viewpoint representation
        self.compose_viewpoints()

    def compose_viewpoints(self):
        """
        Conversion from aligned micro image array to viewpoint array representation. The fundamentals behind the
        4-D light-field transfer were derived by Levoy and Hanrahans in their paper 'Light Field Rendering' in Fig. 6.
        """

        # print status
        self.sta.status_msg('Viewpoint composition', self.cfg.params[self.cfg.opt_prnt])
        self.sta.progress(None, self.cfg.params[self.cfg.opt_prnt])

        # initialize basic light-field parameters
        self._init_vp_img_arr()

        # rearrange light field to multi-view image representation
        for j in range(self._M):
            for i in range(self._M):

                # check interrupt status
                if self.sta.interrupt:
                    return False

                # extract viewpoint by pixel rearrangement
                self._vp_img_arr[j, i] = self._lfp_img_align[j::self._M, i::self._M]

                # print status
                percentage = (j*self._M+i+1)/self._M**2
                self.sta.progress(percentage*100, self.cfg.params[self.cfg.opt_prnt])

        return True

    def decompose_viewpoints(self):
        """
        Conversion from viewpoint image array to aligned micro image array representation. The fundamentals behind the
        4-D light-field transfer were derived by Levoy and Hanrahans in their paper 'Light Field Rendering' in Fig. 6.
        """

        # print status
        self.sta.status_msg('Viewpoint decomposition', self.cfg.params[self.cfg.opt_prnt])
        self.sta.progress(None, self.cfg.params[self.cfg.opt_prnt])

        # initialize basic light-field parameters
        self._init_lfp_img_align()

        # rearrange light field to multi-view image representation
        for j in range(self._M):
            for i in range(self._M):

                # check interrupt status
                if self.sta.interrupt:
                    return False

                # extract viewpoint by pixel rearrangement
                self._lfp_img_align[j::self._M, i::self._M] = self._vp_img_arr[j, i, :, :, :]

                # print status
                percentage = (j*self._M+i+1)/self._M**2
                self.sta.progress(percentage*100, self.cfg.params[self.cfg.opt_prnt])

        return True

# local imports
from plenopticam.cfg import PlenopticamConfig
from plenopticam import misc
from plenopticam.lfp_extractor.lfp_cropper import LfpCropper
from plenopticam.lfp_extractor.lfp_rearranger import LfpRearranger
from plenopticam.lfp_extractor.lfp_exporter import LfpExporter
from plenopticam.lfp_extractor.lfp_contrast import LfpContrast
from plenopticam.lfp_extractor.lfp_outliers import LfpOutliers
from plenopticam.lfp_extractor.lfp_color_eq import LfpColorEqualizer
from plenopticam.lfp_extractor.hex_corrector import HexCorrector
from plenopticam.lfp_extractor.lfp_depth import LfpDepth

import pickle
import os


class LfpExtractorModified(object):

    def __init__(self, lfp_img_align=None, cfg=None, sta=None):

        # input variables
        self._lfp_img_align = lfp_img_align
        self.cfg = cfg if cfg is not None else PlenopticamConfig()
        self.sta = sta if sta is not None else misc.PlenopticamStatus()

        # variables for viewpoint arrays
        self.vp_img_arr = []        # gamma corrected
        self.vp_img_linear = []     # linear gamma (for further processing)
        self.depth_map = None

    def main(self):

        # load previously calculated calibration and aligned data
        self.cfg.load_cal_data()
        if self._lfp_img_align is None:
            self.load_pickle_file()
            self.load_lfp_metadata()

        # micro image crop
        lfp_obj = LfpCropper(lfp_img_align=self._lfp_img_align, cfg=self.cfg, sta=self.sta)
        lfp_obj.main()
        self._lfp_img_align = lfp_obj.lfp_img_align
        del lfp_obj

        # rearrange light-field to sub-aperture images
        if self.cfg.params[self.cfg.opt_view]:
            lfp_obj = LfpRearrangerModified(self._lfp_img_align, cfg=self.cfg, sta=self.sta)
            lfp_obj.main()
            self.vp_img_linear = lfp_obj.vp_img_arr
            del lfp_obj

        # remove outliers if option is set
        if self.cfg.params[self.cfg.opt_lier]:
            obj = LfpOutliers(vp_img_arr=self.vp_img_linear, cfg=self.cfg, sta=self.sta)
            obj.main()
            self.vp_img_linear = obj.vp_img_arr
            del obj

#         # color equalization
#         if self.cfg.params[self.cfg.opt_colo]:
#             obj = LfpColorEqualizer(vp_img_arr=self.vp_img_linear, cfg=self.cfg, sta=self.sta)
#             obj.main()
#             self.vp_img_linear = obj.vp_img_arr
#             del obj

        # copy light-field for refocusing process prior to contrast alignment and export
        self.vp_img_arr = self.vp_img_linear.copy() if self.vp_img_linear is not None else None

#         # color management automation
#         obj = LfpContrast(vp_img_arr=self.vp_img_arr, cfg=self.cfg, sta=self.sta)
#         obj.main()
#         self.vp_img_arr = obj.vp_img_arr
#         del obj

        # reduction of hexagonal sampling artifacts
        if self.cfg.params[self.cfg.opt_arti]:
            obj = HexCorrector(vp_img_arr=self.vp_img_arr, cfg=self.cfg, sta=self.sta)
            obj.main()
            self.vp_img_arr = obj.vp_img_arr
            del obj

        # write viewpoint data to hard drive
        if self.cfg.params[self.cfg.opt_view]:
            obj = LfpExporter(vp_img_arr=self.vp_img_arr, cfg=self.cfg, sta=self.sta)
            obj.write_viewpoint_data()
            del obj

        # compute and write depth data from epipolar analysis
#         if self.cfg.params[self.cfg.opt_dpth]:
#             obj = LfpDepth(vp_img_arr=self.vp_img_arr, cfg=self.cfg, sta=self.sta)
#             obj.main()
#             self.depth_map = obj.depth_map
#             del obj

        return True

    def load_pickle_file(self):
        """ load previously computed light field alignment """

        # file path
        fp = os.path.join(self.cfg.exp_path, 'lfp_img_align.pkl')

        try:
            self._lfp_img_align = pickle.load(open(fp, 'rb'))
        except EOFError:
            os.remove(fp)
        except FileNotFoundError:
            return False

        return True

    def load_lfp_metadata(self):
        """ load LFP metadata settings (for Lytro files only) """

        fname = os.path.splitext(os.path.basename(self.cfg.params[self.cfg.lfp_path]))[0]+'.json'
        fp = os.path.join(self.cfg.exp_path, fname)
        if os.path.isfile(fp):
            json_dict = self.cfg.load_json(fp=fp, sta=None)
            from plenopticam.lfp_reader.lfp_decoder import LfpDecoder
            self.cfg.lfpimg = LfpDecoder().filter_lfp_json(json_dict, settings=self.cfg.lfpimg)

        return True



def lf_decode_sans_save(lfp_path, cal_path, full_sai = True, central_view_extract_dim = 3):
	#Configuration
	cfg = pcam.cfg.PlenopticamConfig()
	cfg.default_values()
	cfg.params[cfg.lfp_path] = lfp_path
	cfg.params[cfg.cal_path] = cal_path
	cfg.params[cfg.opt_cali] = True
	cfg.params[cfg.ptc_leng] = 15
	cfg.params[cfg.cal_meth] = pcam.cfg.constants.CALI_METH[3]
	cfg.params[cfg.opt_cont]=False
	cfg.params[cfg.opt_colo]=False
	cfg.params[cfg.opt_awb_]=False
	cfg.params[cfg.opt_sat_]=False
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
	aligner = LfpAlignerModified(lfp_img, cfg, sta, wht_img)
	ret = aligner.main()
	lfp_img_align = aligner.lfp_img

	#Extracting Sub Aperture Images
	extractor = LfpExtractorModified(lfp_img_align, cfg, sta)
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



	



