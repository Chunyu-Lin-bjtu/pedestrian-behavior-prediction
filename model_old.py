# Toy LSTM sample data

import numpy as np
import csv
import matplotlib.pyplot as plt
import keras


from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import SGD, Adam



# read CSV file
print('reading CSV files ...')
with open('trajectory_filtered_no_veh/bidirection_no_vehicle_3v7_01_traj_ped_1.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    head_processed = 0
    data = []
    for row in spamreader:
        # print(row)
        if head_processed:
            row_current = []
            for i in range(len(row)):
                row_current.append(float(row[i]))
            data.append(row_current)
        else:
            head_processed = 1

# generate x_train, y_train
print('generating training and validation data ... ')
'''
nb_ep = 10
nb_batch_size = 100
LSTM_step = 5
LSTM_size = 150
optimizer = keras.optimizers.Adam(lr=0.001)
loss_type = 'mae'
'''

nb_ep = 2000
nb_batch_sizes = [50, 200]
LSTM_steps = [5, 10, 15]
LSTM_sizes = [50, 100]
learning_rates = [0.0005, 0.001]
loss_type = 'mae'

for nb_batch_size in nb_batch_sizes:
    for LSTM_size in LSTM_sizes:
        for LSTM_step in LSTM_steps:
            for learning_rate in learning_rates:
                print('processing batch', nb_batch_size, 'LSTM step', LSTM_step, 'LSTM size', LSTM_size, '...')
                optimizer = keras.optimizers.Adam(lr=learning_rate)

                x_data_list = []
                y_data_list = []
                offset_data_list = []
                for i in range(len(data)-1): # i: current time step
                    if not i < LSTM_step:
                        one_sample = []
                        for j in range(i - LSTM_step, i):

                            one_sample.append([data[j][3]-data[i - LSTM_step][3], data[j][5]-data[i - LSTM_step][5]])  # x_est, y_est
                        x_data_list.append(one_sample)
                        y_data_list.append([data[j+1][3]-data[i - LSTM_step][3], data[j+1][5]-data[i - LSTM_step][5]])
                        offset_data_list.append([data[i - LSTM_step][3], data[i - LSTM_step][5]])

                x_data = np.array(x_data_list)
                y_data = np.array(y_data_list)
                offset_data = np.array(offset_data_list)

                # split training and validation data
                ratio = 0.9 # train to all data ratio
                cut = round(ratio * len(x_data))

                x_train = x_data[0:cut, :, :]
                x_train_offset = offset_data[0:cut, :]
                y_train = y_data[0:cut, :]

                x_val = x_data[cut:, :, :]
                x_val_offset = offset_data[cut:, :]
                y_val = y_data[cut:, :]

                # model building
                model = Sequential()
                model.add(LSTM(LSTM_size, input_shape=(LSTM_step, 2), return_sequences=True))
                model.add(Dense(2)) # time distributed dense

                history = keras.callbacks.History()


                model.compile(loss=loss_type, optimizer=optimizer)
                model.summary()
                print('model initiated ...')

                #
                print('start training ...')
                model.fit(x_train, y_train,
                          batch_size=nb_batch_size, epochs=nb_ep,
                          validation_data=(x_val, y_val), verbose=2, callbacks=[history] )

                y_val_pred = model.predict(x_val)

                # plot training loss vs. validation loss
                ep = np.arange(0, nb_ep, 1)
                plt.figure()
                plt.plot(ep, history.history["loss"], 'r--', ep, history.history["val_loss"], 'g')
                plt.title('ep '+str(nb_ep)+' ba '+str(nb_batch_size)+' lr '+str(learning_rate)+' LSTM_size '+str(LSTM_size)+' LSTM_step '+str(LSTM_step))
                plt.grid()
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['training', 'validation'], loc='upper right')
                # plt.show()
                plt.savefig('ep_' + str(nb_ep) + '_ba_'+ str(nb_batch_size) +'_lr_' + str(learning_rate) +'_step_' + str(LSTM_step) +'_size_'+str(LSTM_size) + '_losstype_' + loss_type+'_loss.png')

                y_train_pred = model.predict(x_train)

                # plot trajectories
                plt.figure()
                plt.plot(y_train[:,0]+x_train_offset[:,0], y_train[:,1]+x_train_offset[:,1],'.')
                plt.plot(y_train_pred[:, 0]+x_train_offset[:,0], y_train_pred[:, 1]+x_train_offset[:,1],'.')
                plt.plot(y_val[:,0]+x_val_offset[:,0], y_val[:,1]+x_val_offset[:,1],'.')
                plt.plot(y_val_pred[:, 0]+x_val_offset[:,0], y_val_pred[:, 1]+x_val_offset[:,1],'.')
                plt.title('ep ' + str(nb_ep) + ' ba ' + str(nb_batch_size) + ' lr ' + str(
                    learning_rate) + ' LSTM_size ' + str(LSTM_size) + ' LSTM_step ' + str(LSTM_step))
                plt.grid()
                plt.ylabel('y coordinate')
                plt.ylim(25, 100)
                plt.xlabel('x coordinate')
                plt.xlim(1100, 1200)
                plt.legend(['y_train', 'y_train_pred', 'y_val', 'y_val_pred'], loc='upper right')

                # plt.show()
                plt.savefig('ep_' + str(nb_ep) + '_ba_'+ str(nb_batch_size) +'_lr_' + str(learning_rate) +'_step_' + str(LSTM_step) +'_size_'+str(LSTM_size) + '_losstype_' + loss_type+'_trajectories.png')

print('end')
