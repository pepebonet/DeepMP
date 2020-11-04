#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.inception_v3 import InceptionV3


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


def inception_layer(indata, training, scope_str="inception_layer", times=16):
    import pdb;pdb.set_trace()


def incept_net(training, scopestr="inception_net"):

        # input_signal = signals
        
        model = Sequential()
        with tf.compat.v1.variable_scope(scopestr + "conv_layer1"):
            model.add(tf.keras.layers.InputLayer(input_shape=(1, 360)))
            model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=[1, 7], strides=2, padding="same", use_bias=False, name="conv"))
            model.add(tf.keras.layers.BatchNormalization(trainable=True, momentum=0.9))
            model.add(tf.keras.layers.ReLU())
        with tf.compat.v1.variable_scope(scopestr + "maxpool_layer1"):
            model.add(tf.keras.layers.MaxPooling1D(3, strides=2, padding="same", name="maxpool"))
        with tf.compat.v1.variable_scope(scopestr + "conv_layer2"):
            model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding="same", use_bias=False, name="conv2"))
            model.add(tf.keras.layers.BatchNormalization(trainable=training, momentum=0.9))
            model.add(tf.keras.layers.ReLU())
        with tf.compat.v1.variable_scope(scopestr + "conv_layer3"):
            model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False, name="conv3"))
            model.add(tf.keras.layers.BatchNormalization(trainable=training, momentum=0.9))
            model.add(tf.keras.layers.ReLU())
        # import pdb;pdb.set_trace()
        # inception layer x 11
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer1'):
        #     model.add(inception_layer(x, training, scopestr + "1"))
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer2'):
        #     model.add(inception_layer(x, training, scopestr + "2"))
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer3'):
        #     model.add(inception_layer(x, training, scopestr + "3"))
        # with tf.compat.v1.variable_scope(scopestr + 'maxpool_layer2'):
        #     model.add(tf.keras.layers.MaxPooling2D(x, [1, 3], strides=2, padding="same", name="maxpool"))
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer4'):
        #     model.add(inception_layer(x, training, scopestr + "4"))
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer5'):
        #     model.add(inception_layer(x, training, scopestr + "5"))
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer6'):
        #     model.add(inception_layer(x, training, scopestr + "6"))
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer7'):
        #     model.add(inception_layer(x, training, scopestr + "7"))
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer8'):
        #     model.add(inception_layer(x, training, scopestr + "8"))
        # with tf.compat.v1.variable_scope(scopestr + 'maxpool_layer3'):
        #     model.add(tf.keras.layers.MaxPooling2D(x, [1, 3], strides=2, padding="same", name="maxpool"))
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer9'):
        #     model.add(inception_layer(x, training, scopestr + "9"))
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer10'):
        #     model.add(inception_layer(x, training, scopestr + "10"))
        # with tf.compat.v1.variable_scope(scopestr + 'incp_layer11'):
        #     model.add(inception_layer(x, training, scopestr + "11"))
        # with tf.compat.v1.variable_scope(scopestr + 'avgpool_layer1'):
        #     model.add(tf.keras.layers.AveragePooling1D(7, strides=1, padding="same", name="avgpool"))
        
        model.compile(loss='binary_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(),
                   metrics=['accuracy'])
        print(model.summary())
        # x_shape = x.get_shape().as_list()
        # signal_model_output = tf.reshape(x, [-1, x_shape[2] * x_shape[3]])
        return model

