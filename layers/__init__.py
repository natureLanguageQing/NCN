# encoding: utf-8

# author: NLQing
# contact: ygq624576166@163.com


# file: __init__.py
# time: 2019-05-23 14:05

import tensorflow as tf
from tensorflow.python import keras
from NCN.layers.non_masking_layer import NonMaskingLayer
from NCN.layers.att_wgt_avg_layer import AttentionWeightedAverageLayer
from NCN.layers.att_wgt_avg_layer import AttentionWeightedAverage, AttWgtAvgLayer
from NCN.layers.kmax_pool_layer import KMaxPoolingLayer, KMaxPoolLayer, KMaxPooling

L = keras.layers

if __name__ == "__main__":
    print("Hello world")
