#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *


class SequenceCNN(Model):

    def __init__(self, **kwargs):
        super(SequenceCNN, self).__init__(**kwargs)
        self.conv1 = Conv1D(256, 3, activation='relu')
        self.localconv1 = LocallyConnected1D(256, 3, activation='relu')
        self.conv2 = Conv1D(256, 3, activation='relu')
        self.localconv2 = LocallyConnected1D(256, 3, activation='relu')
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


class BCErrorCNN(Model):

    def __init__(self, **kwargs):
        super(BCErrorCNN, self).__init__(**kwargs)
        self.conv1 = Conv1D(128, 3, activation='relu')
        self.localconv1 = LocallyConnected1D(128, 3, activation='relu')
        self.maxpool = MaxPooling1D()
        self.localconv2 = LocallyConnected1D(128, 3, activation='relu')
        self.avgpool = GlobalAveragePooling1D(name='err_pooling_layer')
        self.dense1 = Dense(100, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense2 = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, submodel = False):
        x = self.conv1(inputs)
        x = self.localconv1(x)
        x = self.maxpool(x)
        x = self.localconv2(x)
        x = self.avgpool(x)
        if submodel:
          return x
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


class JointNN(Model):

    def __init__(self, **kwargs):
        super(JointNN, self).__init__(**kwargs)
        self.seqnn = SequenceCNN()
        self.errnn = BCErrorCNN()
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


def get_brnn_model(base_num, embedding_size, features = 5, rnn_cell = "lstm"):

    depth = embedding_size + features
    input = Input(shape=(base_num, depth))

    if rnn_cell == "gru":
        x = Bidirectional(RNN([GRUCell(256, dropout=0.2), \
                GRUCell(256, dropout=0.2),GRUCell(256, dropout=0.2)]))(input)
    else:
        x = Bidirectional(RNN([LSTMCell(256, dropout=0.2), \
                LSTMCell(256, dropout=0.2),LSTMCell(256, dropout=0.2)]))(input)
    #x = Dense(50, activation='relu', use_bias=False)(x)
    x = Dense(512, activation='relu', use_bias=False)(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', use_bias=False)(x)
    model = Model(input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    return model

def get_sequence_model(base_num, embedding_size, features = 4):

    depth = embedding_size + features
    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(base_num,depth)))
    model.add(tf.keras.layers.Conv1D(256, 3, activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(256, 3, activation='relu'))
    model.add(tf.keras.layers.Conv1D(256, 3, activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(256, 3, activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling1D(name='seq_pooling_layer'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False))

    model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    print(model.summary())
    return model

#TODO delete in future
def get_error_model(feat):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(128, 3, padding='same', input_shape=(feat, 1), activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(256, 3, activation='relu'))
    # model.add(tf.keras.layers.MaxPooling1D(2))
    model.add(tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(256, 3, activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling1D(name='err_pooling_layer'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False))

    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['acc']
    )
    print(model.summary())
    return model


def joint_model(base_num, embedding_size):

    model1 = get_sequence_model(base_num, embedding_size)
    output1 = model1.get_layer("seq_pooling_layer").output
    model2 = get_single_err_model(base_num)
    output2 = model2.get_layer("err_pooling_layer").output

    x = concatenate([output1, output2],axis=-1)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', use_bias=False)(x)
    model = Model(inputs=[model1.input, model2.input], outputs=out)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    print(model.summary())

    return model


def get_single_err_model(base_num, depth = 9):

    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(base_num,depth)))
    model.add(tf.keras.layers.Conv1D(128, 3, activation='relu'))
    model.add(tf.keras.layers.LocallyConnected1D(128, 3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.LocallyConnected1D(128, 3, activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling1D(name='err_pooling_layer'))
    #model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False))

    model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    print(model.summary())
    return model