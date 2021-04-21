from .LFRotz import LFRotz
# Change this later!!!
from .trial import maketform, imtransform, SliceXYImage
import numpy as np
import cv2
from scipy.interpolate import interp1d


def LFDecodeLensletImageSimple(LensletImage, WhiteImage, LensletGridModel, DecodeOptions,
                               return_LFWeight=False, return_DecodeOptions=False, return_DebayerLensletImage=False,
                               return_CorrectedLensletImage=False):
    # set default parameters
    DecodeOptions["LevelLimits"] = [np.min(WhiteImage), np.max(WhiteImage)]
    DecodeOptions["ResampMethod"] = "fast"
    DecodeOptions["Precision"] = np.float32
    DecodeOptions["DoDehex"] = True
    DecodeOptions["DoSquareST"] = True
    intmax_uint16 = float(2**16-1)

    # Rescale image values, remove black level
    DecodeOptions["LevelLimits"] = DecodeOptions["Precision"](
        DecodeOptions["LevelLimits"])
    BlackLevel = DecodeOptions["LevelLimits"][0]
    WhiteLevel = DecodeOptions["LevelLimits"][1]
    WhiteImage = WhiteImage.astype(DecodeOptions["Precision"])
    WhiteImage = (WhiteImage - BlackLevel)/(WhiteLevel - BlackLevel)

    LensletImage = LensletImage.astype(DecodeOptions["Precision"])
    LensletImage = (LensletImage - BlackLevel)/(WhiteLevel - BlackLevel)
    LensletImage = LensletImage/WhiteImage
    # Clip -- this is aggressive and throws away bright areas; there is a potential for an HDR approach here
    LensletImage = np.minimum(1, np.maximum(0, LensletImage))

    nargout = 1 + np.sum([return_CorrectedLensletImage,
                          return_DebayerLensletImage, return_DecodeOptions, return_LFWeight])
    if nargout < 2:
        del WhiteImage

    LensletImage = (
        LensletImage*intmax_uint16).astype(DecodeOptions["Precision"])
    # THIS LINE IS NOT SURE TO WORK. FIND SOMETHING CONCRETE THAT WORKS WITH LFS
    LensletImage = cv2.cv2Color(LensletImage, cv2.COLOR_BAYER_BG2RGB)
    LensletImage = LensletImage.astype(DecodeOptions["Precision"])
    LensletImage = LensletImage/intmax_uint16
    DecodeOptions["NColChans"] = 3

    if nargout >= 2:
        DecodeOptions["NWeightChans"] = 1
    else:
        DecodeOptions["NWeightChans"] = 0

    if nargout > 3:
        DebayerLensletImage = LensletImage

    # Tranform to an integer-spaced grid
    print("\nAligning image to lenslet array...")
    InputSpacing = np.array(
        [LensletGridModel["HSpacing"], LensletGridModel["VSpacing"]])
    NewLensletSpacing = np.ceil(InputSpacing)
    # Force even so hex shift is a whole pixel multiple
    NewLensletSpacing = np.ceil(NewLensletSpacing/2)*2
    XformScale = NewLensletSpacing/InputSpacing

    NewOffset = np.array([LensletGridModel["HOffset"],
                          LensletGridModel["VOffset"]]) * XformScale
    RoundedOffset = np.round(NewOffset)
    XformTrans = RoundedOffset-NewOffset

    NewLensletGridModel = {'HSpacing': NewLensletSpacing[0], 'VSpacing': NewLensletSpacing[1],
                           'HOffset': RoundedOffset[0], 'VOffset': RoundedOffset[1], 'Rot': 0,
                           'UMax': LensletGridModel["UMax"], 'VMax': LensletGridModel["VMax"],
                           'Orientation': LensletGridModel["Orientation"],
                           'FirstPosShiftRow': LensletGridModel["FirstPosShiftRow"]}

    RRot = LFRotz(LensletGridModel["Rot"])
    RScale = np.eye(3)
    RScale[0, 0] = XformScale[0]
    RScale[1, 1] = XformScale[1]
    DecodeOptions.OutputScale[:2] = XformScale
    DecodeOptions.OutputScale[2:4] = np.array([1, 2/np.sqrt(3)])

    RTrans = np.eye(3)
    RTrans[-1, :2] = XformTrans

    # Change this later!!
    ''' This part of the code is not complete '''
    FixAll = maketform('affine', RRot@RScale@RTrans)
    temp = XformScale[0]
    NewSize = np.shape(LensletImage[:, :, 0]*XformScale[::-1].reshape(1, 2))
    LensletImage = imtransform(LensletImage, FixAll, 'YData', np.array(
        [1, NewSize[0]]), 'XData', np.array([1, NewSize[1]]))
    if nargout >= 2:
        WhiteImage = imtransform(WhiteImage, FixAll, 'YData', np.array(
            [1, NewSize[0]]), 'XData', np.array([1, NewSize[1]]))
    if nargout >= 4:
        CorrectedLensletImage = LensletImage

    LF = SliceXYImage(NewLensletGridModel, LensletImage,
                      WhiteImage, DecodeOptions)
    del WhiteImage, LensletImage
    ''' Till this much '''

    # Correct for hex grid and resize to square u,v pixels
    LFSize = list(np.shape(LF))
    HexAspect = 2/np.sqrt(3)

    if DecodeOptions["ResampMethod"] == "fast":
        print("\nResampling (1D approximation) to square u,v pixels")
        n_steps = int(np.ceil(LFSize[3]+1))
        NewUVec = HexAspect*np.arange(n_steps)
        NewUVec = NewUVec[:int(np.ceil(LFSize[3]*HexAspect))]
        OrigUSize = LFSize[3]
        LFSize[3] = len(NewUVec)
        # Allocate dest and copy orig LF into it (memory saving vs. keeping both separately)
        LF2 = np.zeros(LFSize, dtype=DecodeOptions["Precision"])
        LF2[:, :, :, :OrigUSize, :] = LF
        LF = LF2
        del LF2

        if DecodeOptions["DoDehex"]:
            ShiftUVec = -0.5+NewUVec
            print('removing hex sampling...')
        else:
            ShiftUVec = NewUVec
            print("...")

        for ColChan in range(np.shape(LF)[4]):
            CurUVec = ShiftUVec
            for RowIter in range(2):
                # Check if this works!!!
                RowIdx = np.mod(
                    NewLensletGridModel["FirstPosShiftRow"] + RowIter, 2) + 1
                ShiftRows = np.squeeze(
                    LF[:, :, RowIdx:-1:2, :OrigUSize, ColChan])
                SliceSize = list(np.shape(ShiftRows))
                SliceSize[3] = len(NewUVec)
                ShiftRows = ShiftRows.reshape(
                    SliceSize[0]*SliceSize[1]*SliceSize[2], np.shape(ShiftRows)[3])
                ShiftRows_func = interp1d(
                    np.arange(np.shape(ShiftRows)[1]), ShiftRows, kind='linear')
                ShiftRows = ShiftRows_func(CurUVec)
                ShiftRows[~np.isfinite(ShiftRows)] = 0
                LF[:, :, RowIdx:-1:2, :,
                    ColChan] = ShiftRows.reshape(SliceSize)
                CurUVec = NewUVec
        del ShiftRows, ShiftRows_func
        DecodeOptions["OutputScale"][2] = DecodeOptions["OutputScale"][2] * HexAspect
    elif DecodeOptions["ResampMethod"] == "triangulation":

        pass
    else:
        print('\nNo valid dehex / resampling selected\n')

    # Resize to square s,t pixels
    # Assumes only a very slight resampling is required, resulting in an identically-sized output light field
    if DecodeOptions["DoSquareST"]:
        print('\nResizing to square s,t pixels using 1D linear interp...')

        ResizeScale = DecodeOptions["OutputScale"][0] / \
            DecodeOptions["OutputScale"][1]
        ResizeDim1 = 0
        ResizeDim2 = 1
        if ResizeScale < 1:
            ResizeScale = 1/ResizeScale
            ResizeDim1 = 1
            ResizeDim2 = 0

        OrigSize = np.shape(LF)[ResizeDim1]
        OrigVec = np.arange(OrigSize) - OrigSize//2
        NewVec = OrigVec/ResizeScale

        OrigDims = np.arange(5)
        OrigDims = np.delete(OrigDims, ResizeDim1)

        UBlkSize = 32
        USize = np.shape(LF)[3]


        

