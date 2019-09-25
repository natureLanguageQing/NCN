from math import log2, floor
from typing import Dict, Any

import tensorflow as tf

from layers import L
from layers.kmax_pool_layer import KMaxPoolingLayer
from tasks.classification.base_model import BaseClassificationModel


def downsample(inputs, pool_type: str = 'max',
               sorted: bool = True, stage: int = 1):  # noqa: A002
    layers_pool = []
    if pool_type == 'max':
        layers_pool.append(
            L.MaxPooling1D(pool_size=3,
                           strides=2,
                           padding='same',
                           name=f'pool_{stage}'))
    elif pool_type == 'k_max':
        k = int(inputs.shape[1].value / 2)
        layers_pool.append(
            KMaxPoolingLayer(k=k,
                             sorted=sorted,
                             name=f'pool_{stage}'))
    elif pool_type == 'conv':
        layers_pool.append(
            L.Conv1D(filters=inputs.shape[-1].value,
                     kernel_size=3,
                     strides=2,
                     padding='same',
                     name=f'pool_{stage}'))
        layers_pool.append(
            L.BatchNormalization())
    elif pool_type is None:
        layers_pool = []
    else:
        raise ValueError(f'unsupported pooling type `{pool_type}`!')

    tensor_out = inputs
    for layer in layers_pool:
        tensor_out = layer(tensor_out)
    return tensor_out


def conv_block(inputs, filters: int, kernel_size: int = 3,
               activation: str = 'linear', shortcut: bool = True):
    layers_conv_unit = []
    layers_conv_unit.append(
        L.BatchNormalization())
    layers_conv_unit.append(
        L.PReLU())
    layers_conv_unit.append(
        L.Conv1D(filters=filters,
                 kernel_size=kernel_size,
                 strides=1,
                 padding='same',
                 activation=activation))
    layers_conv_block = layers_conv_unit * 2

    tensor_out = inputs
    for layer in layers_conv_block:
        tensor_out = layer(tensor_out)

    if shortcut:
        tensor_out = L.Add()([inputs, tensor_out])

    return tensor_out


def resnet_block(inputs, filters: int, kernel_size: int = 3,
                 activation: str = 'linear', shortcut: bool = True,
                 pool_type: str = 'max', sorted: bool = True, stage: int = 1):  # noqa: A002
    tensor_pool = downsample(inputs, pool_type=pool_type, sorted=sorted, stage=stage)
    tensor_out = conv_block(tensor_pool, filters=filters, kernel_size=kernel_size,
                            activation=activation, shortcut=shortcut)
    return tensor_out


class DPCNN_Model(BaseClassificationModel):
    """
    DPCNN的这个实现需要一个明确声明的序列长度。因此，输入的序列应该事先填充或剪切到给定的长度。
    This implementation of DPCNN requires a clear declared sequence length.
    So sequences input in should be padded or cut to a given length in advance.
    """

    @classmethod
    def get_default_hyper_parameters(cls) -> Dict[str, Dict[str, Any]]:
        pool_type = 'max'
        filters = 250
        activation = 'linear'
        return {
            'region_embedding': {
                'filters': filters,
                'kernel_size': 3,
                'strides': 1,
                'padding': 'same',
                'activation': activation,
                'name': 'region_embedding',
            },
            'region_dropout': {
                'rate': 0.2,
            },
            'conv_block': {
                'filters': filters,
                'kernel_size': 3,
                'activation': activation,
                'shortcut': True,
            },
            'resnet_block': {
                'filters': filters,
                'kernel_size': 3,
                'activation': activation,
                'shortcut': True,
                'pool_type': pool_type,
                'sorted': True,
            },
            'dense': {
                'units': 256,
                'activation': activation,
            },
            'dropout': {
                'rate': 0.5,
            },
            'activation': {
                'activation': 'softmax',
            }
        }

    def build_model_arc(self):
        output_dim = len(self.pre_processor.label2idx)
        config = self.hyper_parameters
        embed_model = self.embedding.embed_model

        layers_region = [
            L.Conv1D(**config['region_embedding']),
            L.BatchNormalization(),
            L.PReLU(),
            L.Dropout(**config['region_dropout'])
        ]

        layers_main = [
            L.GlobalMaxPooling1D(),
            L.Dense(**config['dense']),
            L.BatchNormalization(),
            L.PReLU(),
            L.Dropout(**config['dropout']),
            L.Dense(output_dim, **config['activation'])
        ]

        tensor_out = embed_model.output

        # build region tensors
        for layer in layers_region:
            tensor_out = layer(tensor_out)

        # build the base pyramid layer
        tensor_out = conv_block(tensor_out, **config['conv_block'])
        # build the above pyramid layers while `steps > 2`
        seq_len = tensor_out.shape[1].value
        if seq_len is None:
            raise ValueError('`sequence_length` should be explicitly assigned, but it is `None`.')
        for i in range(floor(log2(seq_len)) - 2):
            tensor_out = resnet_block(tensor_out, stage=i + 1,
                                      **config['resnet_block'])
        for layer in layers_main:
            tensor_out = layer(tensor_out)

        self.tf_model = tf.keras.Model(embed_model.inputs, tensor_out)
