import tensorflow as tf
from keras import losses
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Activation, Flatten, Conv2D, Dense, Dropout, LSTM
import datetime
import numpy as np


class MainModel:

    def __init__(self):

        pass

    def expand_dim(self, input):
        return tf.expand_dims(input, -1)

    def model_build(self, module_1, module_2, input_1, input_2):

        output_1 = 0
        output_2 = 0
        if module_1 == 'cnn':
            input_1 = tf.keras.layers.Lambda(self.expand_dim)(input_1)
            # input_1 = tf.expand_dims(input_1, -1)
            output_1 = self.cnn_module(input_1)
        elif module_1 == "lstm":
            output_1 = self.rnn_module(input_1)
        elif module_1 == 'attention':
            output_1 = self.attention_module(input_1)

        if module_2 == 'cnn':
            # input_2 = tf.expand_dims(input_2, -1)
            input_2 = tf.keras.layers.Lambda(self.expand_dim)(input_2)
            # input_2 = tf.expand_dims(input_2, -1)
            output_2 = self.cnn_module(input_2)
        elif module_2 == 'lstm':
            output_2 = self.rnn_module(input_2)
        elif module_2 == 'attention':
            output_2 = self.attention_module(input_2)

        # original_text_emotion_prediction, emotion_cause_prediction = self.output_module(output_1, output_2)
        main_model = self.output_module(output_1, output_2)
        # loss = self.customized_loss(emotion_cause_prediction, emotion_cause_label, original_text_emotion_prediction, emotion_label)
        
        # main_model = tf.keras.Model(inputs=[input_1, input_2, emotion_label, emotion_cause_label],
        #                             outputs=[emotion_cause_prediction, original_text_emotion_prediction])
        # main_model.add_loss(loss)

        return main_model

    def cnn_module(self, input):

        with tf.name_scope("information_confusion_module"):
            conv_1 = Conv2D(8, (5, 5), strides=(2, 2), padding="same", data_format="channels_last",
                            input_shape=(tf.shape(input)[1], tf.shape(input)[2], 1))(input)
            conv_1 = BatchNormalization()(conv_1)
            conv_1 = MaxPooling2D(pool_size=(2, 2), strides=(
                1, 1), padding='same')(conv_1)
            conv_1 = Activation("relu")(conv_1)

            conv_2 = Conv2D(8, (5, 5), strides=(
                2, 2), padding="same", data_format="channels_last")(conv_1)
            conv_2 = BatchNormalization()(conv_2)
            conv_2 = MaxPooling2D(pool_size=(2, 2), strides=(
                1, 1), padding='same')(conv_2)
            conv_2 = Activation("relu")(conv_2)

            conv_3 = Conv2D(16, (10, 10), strides=(
                2, 2), padding="same", data_format="channels_last")(conv_2)
            conv_3 = BatchNormalization()(conv_3)
            conv_3 = MaxPooling2D(pool_size=(2, 2), strides=(
                1, 1), padding='same')(conv_3)
            conv_3 = Activation("relu")(conv_3)

            output = Flatten()(conv_3)
            output = Dense(64, activation='relu')(output)

        return output

    def rnn_module(self, input):

        with tf.name_scope("rnn_information_confusion_module"):
            lstm_1 = tf.keras.layers.Bidirectional(
                LSTM(32, return_sequences=True))(input)
            output = tf.keras.layers.Bidirectional(LSTM(32))(lstm_1)

            return output

    def matmul(self, input):
        return tf.matmul(input[0], input[1])

    def transpose(self, input):
        return tf.transpose(input, [0, 2, 1])

    def attention_module(self, input):

        with tf.name_scope("attention_module"):
            # Linear projections
            d_model = 64
            Q = Dense(d_model, use_bias=False)(input)  # (N, T_q, d_model)
            K = Dense(d_model, use_bias=False)(input)  # (N, T_k, d_model)
            V = Dense(d_model, use_bias=False)(input)  # (N, T_k, d_model)

            K = tf.keras.layers.Lambda(self.transpose)(K)
            output = tf.keras.layers.Lambda(
                self.matmul)([Q, K])  # (N, T_q, T_k)

            # softmax
            output = tf.nn.softmax(output)

            output = tf.keras.layers.Lambda(self.matmul)([output, V])

            output = Flatten()(output)
            output = Dense(64, activation='relu')(output)

            return output

    def concatenate(self, input):
        return tf.concat(input, axis=-1)

    def output_module(self, original_text, event):

        with tf.name_scope("output_module"):

            fc_emo_1 = Dense(64, activation='relu')(original_text)
            fc_emo_1 = Dropout(0.5)(fc_emo_1)

            fc_emo_2 = Dense(7, activation='softmax', name='out_emotion')(fc_emo_1)

            # concatenation_vector = tf.concat([original_text, event], axis=-1)
            concatenation_vector = tf.keras.layers.Lambda(
                self.concatenate)([original_text, fc_emo_2, event])
            # concatenation_vector = tf.keras.layers.Lambda(
            #     self.concatenate)([concatenation_vector, emotion_labels])
            # FC1
            fc_cau_1 = Dense(64, activation='relu')(concatenation_vector)
            fc_cau_1 = Dropout(0.5)(fc_cau_1)

            # FC2
            fc_cau_2 = Dense(2, activation='softmax', name='out_cause')(fc_cau_1)


            return [fc_emo_2, fc_cau_2]
