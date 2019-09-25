# encoding: utf-8
"""
@author: nlqing
@contact: ygq624576166@163.com
@

@version: 1.0
@license: Apache Licence
@file: __init__.py
@time: 2019-05-17 11:15

"""
import os

os.environ['TF_KERAS'] = '1'

import keras_bert
from macros import TaskType

custom_objects = keras_bert.get_custom_objects()
CLASSIFICATION = TaskType.CLASSIFICATION
LABELING = TaskType.LABELING

