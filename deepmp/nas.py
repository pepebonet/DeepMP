#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import deepmp.utils as ut


class ConvBlock(Model):

    def __init__(self, units, filter_size):
        super(ConvBlock, self).__init__()
        self.conv = Conv1D(units, filter_size, padding="same")
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LocalBlock(Model):

    def __init__(self, units, filter_size):
        super(LocalBlock, self).__init__()
        self.localconv = LocallyConnected1D(units, filter_size)
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.localconv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvLocalBlock(Model):

    def __init__(self, units, filter_size):
        super(ConvLocalBlock, self).__init__()
        self.conv = Conv1D(units, filter_size)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.localconv = LocallyConnected1D(units, filter_size)
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

    def __init__(self, cnn_block, block_num, units, filter_size):
        super(SequenceCNN, self).__init__()
        self.block_num = block_num
        if cnn_block == 'conv':
            for i in range(self.block_num):
                setattr(self, "block%i" % i, ConvBlock(units, filter_size))
        elif cnn_block == 'local':
            for i in range(self.block_num):
                setattr(self, "block%i" % i, LocalBlock(units, filter_size))
        else:
            for i in range(self.block_num):
                setattr(self, "block%i" % i, ConvLocalBlock(units, filter_size))
        self.pool = GlobalAveragePooling1D(name='seq_pooling_layer')
        self.dense = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, submodel=False):
        x = inputs
        for i in range(self.block_num):
            x = getattr(self, "block%i" % i)(x)
        x = self.pool(x)
        if submodel:
            return x
        return self.dense(x)


class BCErrorCNN(Model):

    def __init__(self, convlocal_blocks, local_blocks, units, filter_size):
        super(BCErrorCNN, self).__init__()
        self.convlocal_blocks = convlocal_blocks
        self.local_blocks = local_blocks
        #self.block1 = ConvLocalBlock(128, 3)
        for i in range(self.convlocal_blocks):
            setattr(self, "clcblock%i" % i, ConvLocalBlock(units, filter_size))
        self.maxpool = MaxPooling1D()
        #self.block2 = LocalBlock(128, 3)
        for i in range(self.local_blocks):
            setattr(self, "lblock%i" % i, LocalBlock(units, filter_size))
        self.avgpool = GlobalAveragePooling1D(name='err_pooling_layer')
        self.dense1 = Dense(100, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, submodel = False):
        x = inputs
        #x = self.block1(inputs)
        for i in range(self.convlocal_blocks):
            x = getattr(self, "clcblock%i" % i)(x)
        x = self.maxpool(x)
        #x = self.block2(x)
        for i in range(self.local_blocks):
            x = getattr(self, "lblock%i" % i)(x)
        x = self.avgpool(x)
        if submodel:
          return x
        x = self.dense1(x)
        return self.dense2(x)


class JointNN(Model):

    def __init__(self, params):
        super(JointNN, self).__init__()
        self.seqnn = SequenceCNN(params['seq_block_type'], params['seq_block_num'], params['seq_units'], params['seq_filter'])
        self.errnn = BCErrorCNN(params['err_clc'], params['err_lc'], params['err_units'], params['err_filter'])
        self.dense1 = Dense(params['fc_units'], activation='relu')
        self.dropout = Dropout(params['dropout_rate'])
        self.dense2 = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs):
        a = self.seqnn(inputs[0], submodel=True)
        b = self.errnn(inputs[1], submodel=True)
        x = concatenate([a, b], axis=-1)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

def train_jm(train_file, val_file, log_dir, model_dir, batch_size, kmer, epochs, params):

    input_train_seq, input_train_err, label = ut.get_data_jm(train_file, kmer)
    input_val_seq, input_val_err, vy = ut.get_data_jm(val_file, kmer)

    ## train model
    model = JointNN(params)
    model.compile(loss='binary_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(),
                   metrics=['accuracy'])
    input_shape = ([(None, kmer, 9), (None, kmer, 9)])
    model.build(input_shape)
    print(model.summary())

    a = params['seq_block_num']
    b = params['seq_units']
    c = params['seq_filter']
    d = params['err_clc']
    e = params['err_units']
    f = params['err_filter']
    g = params['fc_layers']
    h = params['fc_units']

    log_dir = log_dir + 'test_' + params['seq_block_type'] + '_{}_{}_{}_{}_{}_{}_{}_{}'.format(a,b,c,d,e,f,g,h)
    model_dir = model_dir + 'test_' + params['seq_block_type'] + '_{}_{}_{}_{}_{}_{}_{}_{}'.format(a,b,c,d,e,f,g,h)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
                                            log_dir = log_dir, histogram_freq=1)
    callback_list = [
                        tensorboard_callback,
                        tf.keras.callbacks.ModelCheckpoint(filepath= model_dir,
                                                            monitor='val_accuracy',
                                                            mode='max',
                                                            save_best_only=True,
                                                            save_weights_only= False)
                        ]

    model.fit([input_train_seq, input_train_err], label, batch_size=batch_size, epochs=epochs,
                                                callbacks = callback_list,
                                                validation_data = ([input_val_seq, input_val_err], vy))
    model.save(model_dir)

    return None


t_f = '/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/PRJEB23027/final_features/Norwich/train_test_val_split/reads/train_combined.h5'
v_f = '/cfs/klemming/nobackup/m/mandiche/DeepMP-master/data/PRJEB23027/final_features/Norwich/train_test_val_split/reads/val_combined.h5'

# 'seq_block_type' can choose from ['conv','local','convlocal']

"""for block_type in ['conv','local']:
    for block_num in [2,3,4,5,6]:
        for units in [64,128,256,512,1024]:
            for filter in [3,4,5,6]:

                params = {'seq_block_type': block_type, 'seq_block_num': block_num,
                            'seq_units' : units, 'seq_filter' : filter, 'err_clc' : 1, \
                            'err_lc' : 1, 'err_units' : 128, 'err_filter' : 3,\
                            'fc_layers' : 1, 'fc_units' : 512, 'dropout_rate' : 0.2 }
                try:
                    train_jm(t_f,v_f,'./logs/','./models/',512,17,5, params)
                except:
                    continue
"""

for block_num in [1,2,3,4]:
    for units in [64,128,256,512,1024]:
        for filter in [3,4,5,6]:
            params = {'seq_block_type': 'convlocal', 'seq_block_num': block_num,
                            'seq_units' : units, 'seq_filter' : filter, 'err_clc' : 1, \
                            'err_lc' : 1, 'err_units' : 128, 'err_filter' : 3,\
                            'fc_layers' : 1, 'fc_units' : 512, 'dropout_rate' : 0.2 }
            try:
                train_jm(t_f,v_f,'./logs/','./models/',512,17,5, params)
            except:
                continue
