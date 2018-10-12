import pickle
import cv2
import os
import numpy as np
import csv
from tqdm import tqdm
import pandas as pd


import tensorflow

# test adding a line

class Pedestrian:

    def __init__(self, id_n=1):
        self.id = id_n

        self.trajectory = []

        self.x = []
        self.y = []
        self.v_x = []
        self.v_y = []
        self.frame = []

        self.sur_pedestrian_grid = []
        self.sur_vehicle_grid = []

    def read_trajectory_csv(self, file_path='trajectory_filtered_no_veh/bidirection_no_vehicle_3v7_01_traj_ped_1.csv'):

        print('reading CSV files ...')
        with open(file_path, newline='') as csv_file:
            spam_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            head_processed = 0
            data = []
            for row in spam_reader:
                # print(row)
                if head_processed:
                    row_current = []
                    for i in range(len(row)):
                        row_current.append(float(row[i]))
                    data.append(row_current)
                    self.frame.append(row_current[0])
                    self.x.append(row_current[2])
                    self.y.append(row_current[4])
                    self.v_x.append(row_current[3])
                    self.v_y.append(row_current[5])
                else:
                    head_processed = 1


class DataSet:

    def __init__(self, nb_timesteps=10):

        self.nb_timesteps = nb_timesteps

        self.pedestrians = []

        self.x_data = []
        self.y_data = []
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []

    def add_pedestrian(self, id_n):

        pedestrian = Pedestrian(id_n)
        self.pedestrians.append(pedestrian)

    def split_data(self, train_2_all_ratio=0.9):

        # split training and validation data
        cut = round(train_2_all_ratio * len(self.x_data))

        self.x_train = self.x_data[0:cut, :, :]
        self.y_train = self.y_data[0:cut, :]

        self.x_val = self.x_data[cut:, :, :]
        self.y_val = self.y_data[cut:, :]

    def convert_trajectory_to_input_format(self, nb_timesteps=10):
        # generate x_train, y_train
        x_data_list = []
        y_data_list = []
        for i in range(len(data) - 1):  # i: current time step
            if not i < self.nb_timesteps:
                one_sample = []
                for j in range(i - self.nb_timesteps, i):
                    one_sample.append([data[j][3], data[j][5]])  # x_est, y_est
                x_data_list.append(one_sample)
                y_data_list.append([data[j + 1][3], data[j + 1][5]])

        self.x = np.array(x_data_list)
        self.y = np.array(y_data_list)