from dataset import DataSet, Pedestrian

dataset = DataSet()
dataset.add_pedestrian(1)
dataset.pedestrians[0].read_trajectory_csv()