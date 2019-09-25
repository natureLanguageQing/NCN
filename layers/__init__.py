# encoding: utf-8

# author: nlqing
# contact: ygq624576166@163.com
#

# file: __init__.py
# time: 2019-05-23 14:05

from tensorflow.python import keras

from layers.att_wgt_avg_layer import AttentionWeightedAverageLayer
from layers.non_masking_layer import NonMaskingLayer

L = keras.layers

if __name__ == "__main__":
    print("Hello world")
