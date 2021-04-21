import sys
import numpy as np
# import functions from the LFSign file in the SupportFunctions directory
from SupportFunctions.LFSign import LFSign
from SupportFunctions.LFDefaultField import LFDefaultField
from SupportFunctions.LFStruct2Var import LFStruct2Var
from SupportFunctions.trial import trial
# the line below is to import functions from another directory
# ensure this is there wherever you want to do similar imports
sys.path.append("SupportFunctions")


# something random. 
print(trial(10))