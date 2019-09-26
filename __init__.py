# encoding: utf-8
"""
@author: NLQing
@contact: ygq624576166@163.com


@version: 1.0
@license: Apache Licence
@file: __init__.py
@time: 2019-05-17 11:15

"""
import os
os.environ['TF_KERAS'] = '1'

import keras_bert
from NCN.macros import TaskType, config

custom_objects = keras_bert.get_custom_objects()
CLASSIFICATION = TaskType.CLASSIFICATION
LABELING = TaskType.LABELING

from NCN.version import __version__

from NCN import layers
from NCN import corpus
from NCN import embeddings
from NCN import macros
from NCN import processors
from NCN import tasks
from NCN import utils
from NCN import callbacks

import tensorflow as tf
