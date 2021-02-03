#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3


class ConvBlock(Model):

    def __init__(self, filters, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = Conv1D(filters, kernel_size, padding="same",
                            kernel_initializer="lecun_normal")
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LocalBlock(Model):

    def __init__(self, filters, kernel_size):
        super(LocalBlock, self).__init__()
        self.localconv = LocallyConnected1D(filters, kernel_size,
                                            kernel_initializer="lecun_normal")
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.localconv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvLocalBlock(Model):

    def __init__(self, filters, kernel_size):
        super(ConvLocalBlock, self).__init__()
        self.conv = Conv1D(filters, kernel_size, padding="same",
                            kernel_initializer="lecun_normal")
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.localconv = LocallyConnected1D(filters, kernel_size)
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.localconv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class SequenceCNN(Model):

    def __init__(self, cnn_block, block_num, filters, kernel_size):
        super(SequenceCNN, self).__init__()
        self.block_num = block_num
        if cnn_block == 'conv':
            for i in range(self.block_num):
                setattr(self, "block%i" % i, ConvBlock(filters, kernel_size))
        elif cnn_block == 'local':
            for i in range(self.block_num):
                setattr(self, "block%i" % i, LocalBlock(filters, kernel_size))
        else:
            for i in range(self.block_num):
                setattr(self, "block%i" % i, ConvLocalBlock(filters, kernel_size))
        self.pool = GlobalAveragePooling1D(name='seq_pooling_layer')
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, submodel=False):
        x = inputs
        for i in range(self.block_num):
            x = getattr(self, "block%i" % i)(x)
        x = self.pool(x)
        if submodel:
            return x
        x = self.dense1(x)
        return self.dense2(x)


class BCErrorCNN(Model):

    def __init__(self, conv_blocks, local_blocks, filters, kernel_size):
        super(BCErrorCNN, self).__init__()
        self.conv_blocks = conv_blocks
        self.local_blocks = local_blocks
        for i in range(self.conv_blocks):
            setattr(self, "cblock%i" % i, ConvBlock(filters, kernel_size))
        self.maxpool = MaxPooling1D()
        for i in range(self.local_blocks):
            setattr(self, "lblock%i" % i, LocalBlock(filters, kernel_size))
        self.avgpool = GlobalAveragePooling1D(name='err_pooling_layer')
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, submodel = False):
        x = inputs
        for i in range(self.conv_blocks):
            x = getattr(self, "cblock%i" % i)(x)
        x = self.maxpool(x)
        for i in range(self.local_blocks):
            x = getattr(self, "lblock%i" % i)(x)
        x = self.avgpool(x)
        if submodel:
          return x
        x = self.dense1(x)
        return self.dense2(x)


class JointNN(Model):

    def __init__(self):
        super(JointNN, self).__init__()
        self.seqnn = SequenceCNN('conv', 6, 256, 4)
        self.errnn = BCErrorCNN(3, 3, 128, 3)
        self.dense1 = Dense(512, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense2 = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs):
        a = self.seqnn(inputs[0], submodel=True)
        b = self.errnn(inputs[1], submodel=True)
        x = concatenate([a, b], axis=-1)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


class SequenceBRNN(Model):

    def __init__(self, units, dropout_rate, rnn_cell):
        super(SequenceBRNN, self).__init__()
        if rnn_cell == "gru":
            self.brnn = Bidirectional(RNN([GRUCell(units, dropout=dropout_rate), \
                                GRUCell(units, dropout=dropout_rate),GRUCell(units, dropout=dropout_rate)]))
        else:
            self.brnn = Bidirectional(RNN([LSTMCell(units, dropout=dropout_rate), \
                    LSTMCell(units, dropout=dropout_rate),LSTMCell(units, dropout=dropout_rate)]))
        self.fc1 = Dense(512, activation='relu', use_bias=False)
        self.dropout = Dropout(0.2)
        self.fc2 = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, submodel=False):
        x = self.brnn(inputs)
        if submodel:
            return x
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)

