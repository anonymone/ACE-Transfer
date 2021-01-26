import sys
sys.path.append('./')

import numpy as np

from Coder.ACETransfer import *

transferEngin = transferNASLearningEngin()

network = ACETransfer(code_assist=transferEngin)


