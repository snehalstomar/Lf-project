import numpy as np
import math

#Decode parameters are specified within DecodeOptions
DecodeOptions ={}
DecodeOptions['LevelLimits'] = [np.min(WhiteImage), np.max(WhiteImage)]
DecodeOptions['ResampMethod'] = "fast"
DecodeOptions['Precision'] = np.float32
DecodeOptions['DoDehex'] = True
DecodeOptions['DoSquareST'] = True



def SliceXYImage(LensletGridModel, LensletImage, WhiteImage, DecodeOptions):
	print("\n Slicing Lenslets into LF..")

	USize = LensletGridModel[Umax]
	VSize = LensletGridModel[Vmax]
	MaxSpacing = max(LensletGridModel[HSpacing], LensletGridModel[VSpacing])	
	SSize = MaxSpacing + 1; 
	TSize = MaxSpacing + 1;	

	#Initiating the tensor that will store the actual 5D lightfield
	LF = np.zeros([TSize, SSize, VSize, USize, DecodeOptions[NColChans] + DecodeOptions[NWeightChans]], dtype = DecodeOptions[Precision])
	TVec = np.int16(np.floor(np.arange((-(TSize-1)/2), (((TSize-1)/2)+1))))
	SVec = np.int16(np.floor(np.arange((-(SSize-1)/2), (((SSize-1)/2)+1))))
	VVec = np.int16(np.arange(0,VSize))
	UBlkSize = 32

	for UBlkSize in range(0, USize, UBlkSize):
		UStop = UStart + UBlkSize - 1
		UStop = min(UStop, USize-1)  
		UVec = np.int16(np.arange(UStart, UStop+1))

		tt,ss,vv,uu = np.meshgrid(TVec, SVec, VVec, UVec)
		#---Build indices into 2D image---	
		LFSliceIdxX = LensletGridModel[HOffset] + np.multiply(uu,LensletGridModel[HSpacing]) + ss
		LFSliceIdxY = LensletGridModel[VOffset] + np.multiply(vv,LensletGridModel[VSpacing]) + tt

		HexShiftStart = LensletGridModel[FirstPosShiftRow]
		LFSliceIdxX[:,:,HexShiftStart::2,:] = LFSliceIdxX[:,:,HexShiftStart::2,:] + LensletGridModel[HSpacing]/2;

		#---Lenslet mask in (s,t))x,y and clip at image edges---
		CurSTAspect = DecodeOptions.OutputScale[1]/DecodeOptions.OutputScale[2]
		R = np.square((np.float32(tt) * CurSTAspect)) + np.square(np.float32(ss))
		