# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os
from time import gmtime, strftime
from tensorflow.keras.callbacks import TensorBoard

def make_tensorboard(set_dir_name=''):
  tictoc = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
  directory_name = tictoc
  log_dir = set_dir_name + '_' + directory_name
  log_dir = os.path.join("rec", log_dir)
  os.mkdir(log_dir)
  tensorboard = TensorBoard(log_dir=log_dir)
  return tensorboard
