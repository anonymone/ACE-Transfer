import sys
sys.path.append('./')
import argparse
import numpy as np
import logging
import os

from Evaluator.Utils.surrogate import auto_seq2seq
from Evaluator.Utils.recoder import create_exp_dir

create_exp_dir('./Res/')
create_exp_dir('./Res/nasbench/')
create_exp_dir('./Res/nasbench/seq2seq/')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(message)s")
fh = logging.FileHandler(os.path.join('./Res/nasbench/seq2seq/', 'experiments.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

model = auto_seq2seq('./Res/nasbench/seq2seq/','./Res/nasbench/seq2seq/','./Res/nasbench/seq2seq/')
model.train(epoch=100)