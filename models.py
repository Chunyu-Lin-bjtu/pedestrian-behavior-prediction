from __future__ import print_function

import numpy as np
import random
import keras
from keras.layers import Dense, GRU, Convolution2D, MaxPooling2D, Flatten, Dropout, Activation, LSTM, Conv2D
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam

from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm


class Models:

    def __init__(self, nb_epoch=10, batch_size=64, name='default model'):

        self.model = []
        self.weights = []
        self.model_json = []
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.name = name

        self.history = []
        self.last_Mpercent_epoch_val_loss = []
        self.m_fold_cross_val_results = []

    def build_loss_history(self, X_train, y_train, X_test, y_test):

        class LossHistory(keras.callbacks.Callback):

            def __init__(self, train_x, train_y, val_x, val_y):
                self.train_x = train_x
                self.train_y = train_y
                self.val_x = val_x
                self.val_y = val_y

                self.AUC_train = []
                self.AUC_val = []
                self.train_loss = []
                self.val_loss = []
                self.f1_train = []
                self.f1_val = []

            def make_prediction(self, scores, threshold=0.3):
                return [1 if score >= threshold else 0 for score in scores]

            def on_train_begin(self, logs=None):
                y_pred_train = self.model.predict_proba(self.train_x)
                y_pred_val = self.model.predict_proba(self.val_x)

                self.AUC_train.append(roc_auc_score(self.train_y, y_pred_train))
                self.AUC_val.append(roc_auc_score(self.val_y, y_pred_val))

                self.f1_train.append(f1_score(self.train_y[:,1], self.make_prediction(y_pred_train[:,1])))
                self.f1_val.append(f1_score(self.val_y[:, 1], self.make_prediction(y_pred_val[:, 1])))

            def on_epoch_end(self, epoch, logs={}):
                y_pred_train = self.model.predict_proba(self.train_x)
                y_pred_val = self.model.predict_proba(self.val_x)

                self.AUC_train.append(roc_auc_score(self.train_y, y_pred_train))
                self.AUC_val.append(roc_auc_score(self.val_y, y_pred_val))
                self.train_loss.append(logs.get('loss'))
                self.val_loss.append(logs.get('val_loss'))
                self.f1_train.append(f1_score(self.train_y[:, 1], self.make_prediction(y_pred_train[:, 1])))
                self.f1_val.append(f1_score(self.val_y[:, 1], self.make_prediction(y_pred_val[:, 1])))

        self.history = LossHistory(X_train, y_train, X_test, y_test)

    def train_model(self, X_train, y_train, X_test, y_test, print_option=0, verbose=2):

        self.build_loss_history(X_train, y_train, X_test, y_test)
        self.model.fit(X_train, y_train,
                       batch_size=self.batch_size,
                       nb_epoch=self.nb_epoch,
                       validation_data=(X_test, y_test), verbose=verbose, callbacks=[self.history])
        self.get_lastMpercent_loss()

        if print_option == 1:
            print(self.last_Mpercent_epoch_val_loss)

    def get_lastMpercent_loss(self, m=0.1):

        index = int(self.nb_epoch*m)
        self.last_Mpercent_epoch_val_loss = sum(self.history.AUC_val[index:])/len(self.history.AUC_val[index:])

    def plot_auc(self, epoch_resolution=100, option='AUC_v_epoch'):

        if option == 'AUC_v_epoch':
            ep = np.arange(0, self.nb_epoch + 1, epoch_resolution)
            plt.plot(ep, self.history.AUC_train[0::epoch_resolution], 'r--', ep, self.history.AUC_val[0::epoch_resolution], 'g')
            plt.show()
        elif option == 'loss_v_epoch':
            plt.plot(self.history.train_loss)
            plt.plot(self.history.val_loss)
            plt.show()
        else:
            ep = np.arange(0, self.nb_epoch + 1, epoch_resolution)
            plt.plot(ep, self.history.AUC_train[0::epoch_resolution], 'r--', ep, self.history.AUC_val[0::epoch_resolution], 'g')
            plt.show()

            plt.plot(self.history.train_loss)
            plt.plot(self.history.val_loss)
            plt.show()

    # Build or load a neural network model. Works with Keras implementations.

    def load_model_json(self, load_path='default_path'):
        # todo
        self.model = []

    def save_model_json(self, save_path='default_path'):
        # todo
        self.model = []

    def build_LSTM_model(self, input_shape, optimizer=Adam(lr=1e-6, decay=1e-5)):

        model = Sequential()

        model.add(GRU(100, return_sequences=False, name="lstm_layer", input_shape=input_shape))
        model.add(Dense(2))

        model.compile(loss='mae', optimizer=optimizer)

        self.model = model
