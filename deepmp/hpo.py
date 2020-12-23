#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
#from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import *
import re
import deepmp.utils as ut



class ConvBlock(Model):

    def __init__(self, units, filters, lambd):
        super(ConvBlock, self).__init__()
        self.conv = Conv1D(units, filters, padding="same",
                            kernel_initializer="he_normal",
                            kernel_regularizer=tf.keras.regularizers.l2(lambd))
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LocalBlock(Model):

    def __init__(self, units, filters, lambd):
        super(LocalBlock, self).__init__()
        self.localconv = LocallyConnected1D(units, filters,
                                            kernel_initializer="he_normal",
                                            kernel_regularizer=tf.keras.regularizers.l2(lambd))
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs):
        x = self.localconv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvLocalBlock(Model):

    def __init__(self, units, filters, lambd):
        super(ConvLocalBlock, self).__init__()
        self.conv = Conv1D(units, filters, padding="same",
                            kernel_initializer="he_normal",
                            kernel_regularizer=tf.keras.regularizers.l2(lambd))
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.localconv = LocallyConnected1D(units, filters)
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

    def __init__(self, cnn_block, block_num, units, filters, lambd):
        super(SequenceCNN, self).__init__()
        self.block_num = block_num
        if cnn_block == 'conv':
            for i in range(self.block_num):
                setattr(self, "block%i" % i, ConvBlock(units, filters, lambd))
        elif cnn_block == 'local':
            for i in range(self.block_num):
                setattr(self, "block%i" % i, LocalBlock(units, filters, lambd))
        else:
            for i in range(self.block_num):
                setattr(self, "block%i" % i, ConvLocalBlock(units, filters, lambd))
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

"""
class BCErrorCNN(Model):

    def __init__(self, convlocal_blocks, local_blocks, units, filters):
        super(BCErrorCNN, self).__init__()
        self.convlocal_blocks = convlocal_blocks
        self.local_blocks = local_blocks
        for i in range(self.convlocal_blocks):
            setattr(self, "clcblock%i" % i, ConvLocalBlock(units, filters))
        self.maxpool = MaxPooling1D()
        for i in range(self.local_blocks):
            setattr(self, "lblock%i" % i, LocalBlock(units, filters))
        self.avgpool = GlobalAveragePooling1D(name='err_pooling_layer')
        self.dense1 = Dense(100, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, submodel = False):
        x = inputs
        for i in range(self.convlocal_blocks):
            x = getattr(self, "clcblock%i" % i)(x)
        x = self.maxpool(x)
        for i in range(self.local_blocks):
            x = getattr(self, "lblock%i" % i)(x)
        x = self.avgpool(x)
        if submodel:
          return x
        x = self.dense1(x)
        return self.dense2(x)
"""

class BCErrorCNN(Model):

    def __init__(self, conv_blocks, local_blocks, units, filters, lambd):
        super(BCErrorCNN, self).__init__()
        self.conv_blocks = conv_blocks
        self.local_blocks = local_blocks
        for i in range(self.conv_blocks):
            setattr(self, "cblock%i" % i, ConvBlock(units, filters, lambd))
        self.maxpool = MaxPooling1D()
        for i in range(self.local_blocks):
            setattr(self, "lblock%i" % i, LocalBlock(units, filters,lambd))
        self.avgpool = GlobalAveragePooling1D(name='err_pooling_layer')
        self.dense1 = Dense(100, activation='relu')
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

    def __init__(self, params):
        super(JointNN, self).__init__()
        self.seqnn = SequenceCNN('conv', 6, 256, 4, params['reg'])
        self.errnn = BCErrorCNN(3, 3, 128, 3, params['reg'])
        self.dense1 = Dense(512, activation='relu')
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
                   optimizer=tf.keras.optimizers.Adam(learning_rate= params['learning_rate']),
                   metrics=['accuracy'])
    input_shape = ([(None, kmer, 9), (None, kmer, 9)])
    model.build(input_shape)
    print(model.summary())

    a = params['learning_rate']
    a = re.sub(r'(?<=\d)[,\.]','',str(a))
    b = params['reg']
    b = re.sub(r'(?<=\d)[,\.]','',str(b))
    c = params['dropout_rate']
    d = batch_size

    log_dir = log_dir + 'hpo_' +  '_{}_{}_{}_{}'.format(a,b,c,d)
    model_dir = model_dir + 'hpo_' +  '_{}_{}_{}_{}'.format(a,b,c,d)

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


#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#                                                            initial_learning_rate=1e-2,
#                                                            decay_steps=10000,
#                                                            decay_rate=0.9)


dropouts = 0.2
batch_size = 512

for _ in range(50):
    r1 = -4 * np.random.rand()
    alpha = 10 ** r1
    alpha = np.around(alpha,4)

    r2 = -4 * np.random.rand()
    lambd = 10 ** r2
    lambd = np.around(lambd,4)
    #lambd = 0.01

    params = {'learning_rate': alpha, 'reg': lambd, 'dropout_rate' : dropouts }

    train_jm(t_f,v_f,'./logs/','./models/',batch_size,17,5,params)
