import pandas as pd

class KDD:
	data_file = ''
	feature_file =''

	def __init__(self, config):
		self.data_file = config['data']
		self.feature_file = config['features']
		self.features = self.get_features()
		self.feature_names = self.get_feature_keys()
		self.feature_types = self.get_feature_types()
		self.dataframe = self.get_data()
		self.data = self.dataframe.drop('class', 1)
		self.categorical_to_cont()
		self.labels = self.dataframe['class']

	def get_feature_keys(self):
		return [x[0] for x in self.features]

	def get_feature_types(self):
		return [x[1] for x in self.features]

	def get_features(self):
		f = open(self.feature_file, 'r')
		features = []
		for x in f.readlines():
			y = x.replace('.\n', '').replace(' ','').split(':')
			features.append(y)
		return features

	def get_data(self):
		names = self.feature_names.append('class')
		return pd.read_csv(self.data_file, header=None, names=self.feature_names)
	
	def get_feature(self, f_name):
		return self.dataframe[f_name]

	def categorical_to_cont(self):
		c_types = self.data.select_dtypes(include=['object']).columns
		for i in list(c_types):
			self.data[i] = pd.Categorical(self.data[i]).codes

	def get_normalized_data(self):
		data_nm = (self.data - self.data.min())/(self.data.max() - self.data.min())
		return data_nm
	
	def get_standardized_data(self):
		data_std = (self.data - self.data.mean())/self.data.std()
		return data_std
	
