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


class CentralCNN(Model):

    def __init__(self, **kwargs):
        super(CentralCNN, self).__init__(**kwargs)
        self.conv1 = Conv1D(256, 3, activation='relu', padding='same')
        self.localconv1 = LocallyConnected1D(256, 1, activation='relu')
        self.conv2 = Conv1D(256, 3, activation='relu', padding='same')
        self.localconv2 = LocallyConnected1D(256, 1, activation='relu')
        self.pool = GlobalAveragePooling1D(name='seq_pooling_layer')
        self.dropout = Dropout(0.2)
        self.dense = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, submodel=False):
        x = self.conv1(inputs)
        x = self.localconv1(x)
        x = self.conv2(x)
        x = self.localconv2(x)
        x = self.pool(x)
        if submodel:
            return x
        x = self.dropout(x)
        return self.dense(x)



class RawSigNN(Model):

    def __init__(self, **kwargs)):
        super(RawSigNN, self).__init__(**kwargs)
        self.brnn = Bidirectional(RNN([GRUCell(256, dropout=0.2), \
                            GRUCell(256, dropout=0.2),GRUCell(256, dropout=0.2)]))
        self.fc = Dense(1, activation='sigmoid',use_bias=False)

    def call(self, inputs, submodel=False):
        x = self.brnn(inputs)
        return self.fc(x)


class Inception(Model):

    def __init__(self, **kwargs):
        super(Inception, self).__init__(**kwargs)
        times = 16
        self.max_pool = MaxPooling1D(3, strides=1,  padding="same")
        self.conv1a_1 = Conv1D(filters=int(times * 3), kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv1a_2 = BatchNormalization(trainable=kwargs['trainable'])
        self.conv1a_3 = ReLU()

        self.conv0b_1 = Conv1D(filters=times * 3, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv0b_2 = BatchNormalization(trainable=kwargs['trainable'])
        self.conv0b_3 = ReLU()

        self.conv0c_1 = Conv1D(filters=times * 2, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv0c_2 = BatchNormalization(trainable=kwargs['trainable'])
        self.conv0c_3 = ReLU()
        self.conv1c_1 = Conv1D(filters=times * 3, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.conv1c_2 = BatchNormalization(trainable=kwargs['trainable'])
        self.conv1c_3 = ReLU()

        self.conv0d_1 = Conv1D(filters=times * 2, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv0d_2 = BatchNormalization(trainable=kwargs['trainable'])
        self.conv0d_3 = ReLU()
        self.conv1d_1 = Conv1D(filters=times * 3, kernel_size=5, strides=1, padding="same", use_bias=False)
        self.conv1d_2 = BatchNormalization(trainable=kwargs['trainable'])
        self.conv1d_3 = ReLU()

        self.conv_stem_1 = Conv1D(filters=times * 3, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv_stem_2 = BatchNormalization(trainable=kwargs['trainable'])

        self.conv0e_1 = Conv1D(filters=times * 2, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv0e_2 = BatchNormalization(trainable=kwargs['trainable'])
        self.conv0e_3 = ReLU()
        self.conv1e_1 = Conv1D(filters=times*4, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.conv1e_2 = BatchNormalization(trainable=kwargs['trainable'])
        self.conv1e_3 = ReLU()
        self.conv2e_1 = Conv1D(filters=times * 3, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv2e_2 = BatchNormalization(trainable=kwargs['trainable'])

        self.conv_plus = ReLU()


    def call(self, inputs, submodel=False):
        x = self.max_pool(inputs)
        x = self.conv1a_1(x)
        x = self.conv1a_2(x)
        x = self.conv1a_3(x)

        y = self.conv0b_1(inputs)
        y = self.conv0b_2(y)
        y = self.conv0b_3(y)

        z = self.conv0c_1(inputs)
        z = self.conv0c_2(z)
        z = self.conv0c_3(z)
        z = self.conv1c_1(z)
        z = self.conv1c_2(z)
        z = self.conv1c_3(z)

        r = self.conv0d_1(inputs)
        r = self.conv0d_2(r)
        r = self.conv0d_3(r)
        r = self.conv1d_1(r)
        r = self.conv1d_2(r)
        r = self.conv1d_3(r)

        s = self.conv_stem_1(inputs)
        s = self.conv_stem_2(s)

        t = self.conv0e_1(inputs)
        t = self.conv0e_2(t)
        t = self.conv0e_3(t)
        t = self.conv1e_1(t)
        t = self.conv1e_2(t)
        t = self.conv1e_3(t)
        t = self.conv2e_1(t)
        t = self.conv2e_2(t)

        u =  tf.math.add(s, t)
        u = self.conv_plus(u)

        return tf.concat([x, y, z, r, u], axis=-1)



class InceptNet(Model):

    def __init__(self, **kwargs):
        super(InceptNet, self).__init__(**kwargs)
        self.conv1_1 = Conv1D(filters=64, kernel_size=7, strides=2, padding="same", use_bias=False)
        self.batch_norm_1 = BatchNormalization(trainable=kwargs['trainable'])
        self.relu = ReLU()
        self.maxpooling = MaxPooling1D(3, strides=2, padding="same")
        self.conv1_2 = Conv1D(filters=128, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.batch_norm_2 = BatchNormalization(trainable=kwargs['trainable'])
        self.relu = ReLU()
        self.conv1_3 = Conv1D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.batch_norm_3 = BatchNormalization(trainable=kwargs['trainable'])
        self.relu = ReLU()
        self.incep_layer_1 = Inception(trainable=kwargs['trainable'])
        self.incep_layer_2 = Inception(trainable=kwargs['trainable'])
        self.incep_layer_3 = Inception(trainable=kwargs['trainable'])
        self.maxpooling_2 = MaxPooling1D(3, strides=2, padding="same")
        self.incep_layer_4 = Inception(trainable=kwargs['trainable'])
        self.incep_layer_5 = Inception(trainable=kwargs['trainable'])
        self.incep_layer_6 = Inception(trainable=kwargs['trainable'])
        self.incep_layer_7 = Inception(trainable=kwargs['trainable'])
        self.incep_layer_8 = Inception(trainable=kwargs['trainable'])
        self.maxpooling_3 = MaxPooling1D(3, strides=2, padding="same")
        self.incep_layer_9 = Inception(trainable=kwargs['trainable'])
        self.incep_layer_10 = Inception(trainable=kwargs['trainable'])
        self.incep_layer_11 = Inception(trainable=kwargs['trainable'])
        self.AveragePooling = AveragePooling1D(7, strides=1, padding="same")
        self.dense_1 = Dense(units=100, use_bias=False)
        self.dropout = Dropout(0.2)
        self.dense_2 = Dense(1, activation='sigmoid', use_bias=False)


    def call(self, inputs, submodel=False):
        x = self.conv1_1(inputs)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        x = self.maxpooling(x)
        x = self.conv1_2(x)
        x = self.batch_norm_2(x)
        x = self.relu(x)
        x = self.conv1_3(x)
        x = self.batch_norm_3(x)
        x = self.relu(x)
        x = self.incep_layer_1(x)
        x = self.incep_layer_2(x)
        x = self.incep_layer_3(x)
        x = self.maxpooling_2(x)
        x = self.incep_layer_4(x)
        x = self.incep_layer_5(x)
        x = self.incep_layer_6(x)
        x = self.incep_layer_7(x)
        x = self.incep_layer_8(x)
        x = self.maxpooling_3(x)
        x = self.incep_layer_9(x)
        x = self.incep_layer_10(x)
        x = self.incep_layer_11(x)
        x = self.AveragePooling(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        return self.dense_2(x)
