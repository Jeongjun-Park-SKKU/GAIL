import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from msvcrt import kbhit
from tabnanny import verbose
import pandas as pd
import sys
import numpy as np
import glob
import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import tensorflow as tf
import time
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import inspect
from typing import List
from keras import backend as K, Model, Input, optimizers
import tensorflow as tf
from sklearn.metrics import mean_squared_error
#from . import TCN, tcn_full_summary
from tcn import TCN, tcn_full_summary
from sklearn.metrics import mean_squared_error
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Conv1D, BatchNormalization, Activation, ReLU, Add, Flatten, concatenate ,multiply
from keras.models import Model, load_model
import tensorflow_addons as tfa
import csv
import onnxmltools
from keras.models import load_model
import glob
import os
import math
import random
import pydot
from keras.utils import plot_model
from attention import Attention
from silence_tensorflow import silence_tensorflow
# python -m tf2onnx.convert --saved-model policy_model_1 --output model_1108.onnnx

silence_tensorflow()
with tf.device('/GPU:0'):

    # 데이터 로딩
    dataframe = pd.read_csv('1018.csv')
    dataframe = dataframe.iloc[2::5, :]
    expert_data = dataframe[['PositionX', 'PositionY', 'PositionZ']].values


    def create_dataset(data, time_steps=10):
        X, Y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps)])
            Y.append(data[i + time_steps])
        return np.array(X), np.array(Y)

    time_steps = 30
    X, Y = create_dataset(expert_data, time_steps)

    print(X.shape)
    print(Y.shape)
    # Loss 값을 저장하기 위한 리스트
    d_losses = []
    p_losses = []


    # 하이퍼파라미터
    learning_rate = 0.001
    epochs = 500

    # 정책 네트워크
    class PolicyNetwork(tf.keras.Model):
        def __init__(self):
            super(PolicyNetwork, self).__init__()
            self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
            self.lstm2 = tf.keras.layers.LSTM(128)
            self.output_layer = tf.keras.layers.Dense(3) # x, y, z 출력
            
        def call(self, x):
            x = self.lstm(x)
            x = self.lstm2(x)
            return self.output_layer(x)

    policy = PolicyNetwork()

    # 판별기
    class Discriminator(tf.keras.Model):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
            self.lstm2 = tf.keras.layers.LSTM(128)
            self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
            
        def call(self, x):
            x = self.lstm(x)
            x = self.lstm2(x)
            return self.output_layer(x)

    discriminator = Discriminator()

    policy_optimizer = tf.keras.optimizers.Adam(learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
        with tf.GradientTape() as p_tape, tf.GradientTape() as d_tape:
            # 에이전트의 행동 예측
            agent_actions = policy(X)
            agent_actions_seq = tf.expand_dims(agent_actions, 1)

            # 판별기를 사용한 전문가와 에이전트의 행동 비교
            expert_score = discriminator(X)
            agent_score = discriminator(agent_actions_seq)
            
            # 손실 계산
            d_loss = -tf.reduce_mean(tf.math.log(expert_score) + tf.math.log(1.0 - agent_score))
            p_loss = -tf.reduce_mean(tf.math.log(agent_score))

            d_losses.append(d_loss.numpy())
            p_losses.append(p_loss.numpy())
            
        # 그래디언트 계산 및 적용
        discriminator_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
        policy_gradients = p_tape.gradient(p_loss, policy.trainable_variables)
        
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
        policy_optimizer.apply_gradients(zip(policy_gradients, policy.trainable_variables))
        
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss.numpy()}, P Loss: {p_loss.numpy()}")

    policy.save('policy_model_1', save_format='tf')
    #policy.save_weights('policy_weights.h5')


    # 그래프 그리기
    plt.figure(figsize=(10,5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title("Discriminator loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('D_loss.png', dpi = 500)
    plt.cla()

    plt.figure(figsize=(10,5))
    plt.plot(p_losses, label='Policy/Generator Loss')
    plt.title("Generator loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('G_loss.png', dpi = 500)
    plt.cla()


