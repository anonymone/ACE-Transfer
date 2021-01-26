import sys
sys.path.append('./')

import numpy as np

from Coder.ACETransfer import *

transferEngin = transferNASLearningEngin()

knowledgeBase = list()
with open("./preData/preNetData.txt") as f:
    for sample in f.readlines():
        [architecture, parameter, error] = sample.split(",")
        #architecture = architecture.replace("<--->","-").split("-")
        #architecture = np.array([np.array([int(i[0]), int(i[1]), int(i[2])]) for i in [unit.split(".") for unit in architecture]])
        parameter = float(parameter.replace("M",""))
        error = float(error)
        knowledgeBase.append((architecture, parameter, error))

transferEngin.preLearning(knowledgeBase=knowledgeBase, knowledgeDimension=2)

network = ACETransfer(code_assist=transferEngin)
print(network.get_dec())


