import numpy as np
import os
import pandas as pd


class DataRecord:
	""" Class for processing single record of sensor data
	"""
	def __init__(self, data_path, meta_path) -> None:
		self.data_path = data_path
		self.meta_path = meta_path

		self.data = self.read_data().drop(['time'], axis=1)
		if 'seconds_elapsed' in self.data.columns:
			self.data.drop(['seconds_elapsed'], axis=1)
		self.data = self.data[['acc_z', 'acc_y', 'acc_x', 'gyr_z', 'gyr_y', 'gyr_x', 'gra_z', 'gra_y', 'gra_x', 'mag_z', 'mag_y', 'mag_x', 'ori_qz', 'ori_qy', 'ori_qx' , 'ori_qw', 'ori_roll', 'ori_pitch', 'ori_yaw']].fillna(0)
		self.meta = self.read_meta()

		self.freq = self.extract_freq()

		self.subject, self.platform, self.label, self.location, self.trial = self.extract_info()
	
	def read_data(self):
		return pd.read_csv(self.data_path)

	def read_meta(self):
		return pd.read_csv(self.meta_path)

	def extract_info(self):
		filename = os.path.basename(self.data_path).split('.')[0]
		splitted_filename = filename.split('_')
		subject = splitted_filename[0]
		devices = splitted_filename[1]
		label = splitted_filename[2]
		trial = splitted_filename[-1]
		if len(splitted_filename) == 4:
			location = 'none'
		else:
			location = splitted_filename[3]

		return subject, devices, label, location, trial
	
	def extract_freq(self):
		return 1000 / int(self.meta.sampleRateMs[0].split('|')[0])

	def get_mean_std(self):
		"""Computes mean and std per sensor channel

		Returns
		-------
		mean : np.ndarray
			mean per channel
		std : np.ndarray
			std per channel
		"""
		mean = self.data.mean(axis=0, skipna=False)	
		std = self.data.std(axis=0, skipna=False)	
		return mean, std

	def sample(self, len_tw_sec, overlap_ratio=0):
		"""_Sample data record into shorter time-windows of the given length

		Parameters
		----------
		len_tw_sec : int
			length of time-windows (seconds) 
		overlap_ratio : float
			overlapping ratio (0 -- no overlap, 0.5 -- half overlapping time-windows)

		Returns
		-------
		numpy.ndarray
			array of sampled time-windows
		"""
		data_array = np.array(self.data.iloc[1:])
		num_entries = int(self.freq / len_tw_sec)
		step = num_entries * (1 - overlap_ratio)
		assert 0 <= overlap_ratio < 1, "Overlap ratio should be in [0, 1)"
		sampled_record = [data_array[i : i + num_entries] for i in range(0, len(data_array), step)]
		if sampled_record[-1].shape != sampled_record[0].shape:
			sampled_record = sampled_record[:-1]
		assert sampled_record[0].shape[0] == num_entries, 'Record data is too small for the analysis'
		return np.stack(sampled_record)


def main():
	data_path = './data/ride_data/dmitrii_android_cycle_bag_1/dmitrii_android_cycle_bag_1.csv'
	meta_path = './data/ride_data/dmitrii_android_cycle_bag_1/meta_dmitrii_android_cycle_bag_1.csv'

	data_record = DataRecord(data_path, meta_path)
	
	print(data_record.data.shape)
	print(data_record.data.head())

	print(data_record.extract_info())
	print(data_record.extract_freq())

	print(data_record.get_mean_std())

	sampled_arr = data_record.sample(len_tw_sec=1, overlap_ratio=0)
	print(sampled_arr.shape)


if __name__ == '__main__':
	main()