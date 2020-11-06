#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3


class SequenceCNN(Model):

    def __init__(self, **kwargs):
        super(SequenceCNN, self).__init__(**kwargs)
        self.conv1 = Conv1D(256, 3)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.localconv1 = LocallyConnected1D(256, 3)
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.conv2 = Conv1D(256, 3)
        self.bn3 = BatchNormalization()
        self.relu3 = ReLU()
        self.localconv2 = LocallyConnected1D(256, 3)
        self.bn4 = BatchNormalization()
        self.relu4 = ReLU()
        self.pool = GlobalAveragePooling1D(name='seq_pooling_layer')
        #self.dropout = Dropout(0.2)
        self.dense = Dense(1, activation='sigmoid', use_bias=False)

    def call(self, inputs, submodel=False):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.localconv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.localconv2(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool(x)
        if submodel:
            return x
        #x = self.dropout(x)
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


class Inception(Model):

    def __init__(self, **kwargs):
        super(Inception, self).__init__(**kwargs)
        times = 16
        self.max_pool = MaxPooling1D(3, strides=1,  padding="same")
        self.conv1a_1 = Conv1D(filters=int(times * 3), kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv1a_2 = BatchNormalization(trainable=True)
        self.conv1a_3 = ReLU()

        self.conv0b_1 = Conv1D(filters=times * 3, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv0b_2 = BatchNormalization(trainable=True)
        self.conv0b_3 = ReLU()

        self.conv0c_1 = Conv1D(filters=times * 2, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv0c_2 = BatchNormalization(trainable=True)
        self.conv0c_3 = ReLU()
        self.conv1c_1 = Conv1D(filters=times * 3, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.conv1c_2 = BatchNormalization(trainable=True)
        self.conv1c_3 = ReLU()

        self.conv0d_1 = Conv1D(filters=times * 2, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv0d_2 = BatchNormalization(trainable=True)
        self.conv0d_3 = ReLU()
        self.conv1d_1 = Conv1D(filters=times * 3, kernel_size=5, strides=1, padding="same", use_bias=False)
        self.conv1d_2 = BatchNormalization(trainable=True)
        self.conv1d_3 = ReLU()

        self.conv_stem_1 = Conv1D(filters=times * 3, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv_stem_2 = BatchNormalization(trainable=True)

        self.conv0e_1 = Conv1D(filters=times * 2, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv0e_2 = BatchNormalization(trainable=True)
        self.conv0e_3 = ReLU()
        self.conv1e_1 = Conv1D(filters=times*4, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.conv1e_2 = BatchNormalization(trainable=True)
        self.conv1e_3 = ReLU()
        self.conv2e_1 = Conv1D(filters=times * 3, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.conv2e_2 = BatchNormalization(trainable=True)

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
        self.batch_norm_1 = BatchNormalization(trainable=True)
        self.relu = ReLU()
        self.maxpooling = MaxPooling1D(3, strides=2, padding="same")
        self.conv1_2 = Conv1D(filters=128, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.batch_norm_2 = BatchNormalization(trainable=True)
        self.relu = ReLU()
        self.conv1_3 = Conv1D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.batch_norm_3 = BatchNormalization(trainable=True)
        self.relu = ReLU()
        self.incep_layer_1 = Inception()
        self.incep_layer_2 = Inception()
        self.incep_layer_3 = Inception()
        self.maxpooling_2 = MaxPooling1D(3, strides=2, padding="same")
        self.incep_layer_4 = Inception()
        self.incep_layer_5 = Inception()
        self.incep_layer_6 = Inception()
        self.incep_layer_7 = Inception()
        self.incep_layer_8 = Inception()
        self.maxpooling_3 = MaxPooling1D(3, strides=2, padding="same")
        self.incep_layer_9 = Inception()
        self.incep_layer_10 = Inception()
        self.incep_layer_11 = Inception()
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
