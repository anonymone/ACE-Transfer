import numpy as np
import pandas as pd
from pandas import DataFrame
import random
from copy import deepcopy
import uuid
import logging

from Coder import ACE

class ACETransfer(ACE):
    def __init__(self,
                 fitness_size=2,
                 unit_number_range=(10, 20),
                 value_boundary=(0, 15),
                 classes=10,
                 layers=6,
                 channels=48,
                 keep_prob=0.8,
                 drop_path_keep_prob=0.8,
                 use_aux_head=True,
                 **kwargs):
        super(ACETransfer, self).__init__(
            fitness_size = fitness_size,
            unit_number_range = unit_number_range,
            value_boundary = value_boundary,
            classes = classes,
            layers = layers,
            channels = channels,
            keep_prob = keep_prob,
            drop_path_keep_prob = drop_path_keep_prob,
            use_aux_head = use_aux_head
        )
        for name in kwargs.keys():
            exec("self.{0} = kwargs['{0}']".format(str(name)))

    def code_init(self, codeAssist=None):
        if codeAssist is not None:
            dec = np.array([
                codeAssist.getTop(10)[random.randint(0,10)]
                if random.random() < 0.8
                else [np.random.randint(0, 4) if random.random() < 0.8 else np.random.randint(0, 7), np.random.randint(
            low=self.vb[0], high=self.vb[1]), np.random.randint(low=self.vb[0], high=self.vb[1])]
            for _ in range(random.randint(self.unr[0], self.unr[1]))
            ])
        else:
            dec = np.array([[np.random.randint(0, 4) if random.random() < 0.8 else np.random.randint(0, 7), np.random.randint(
            low=self.vb[0], high=self.vb[1]), np.random.randint(low=self.vb[0], high=self.vb[1])] for i in range(random.randint(self.unr[0], self.unr[1]))])
            logging.error("codeAssist is None!")
        return dec