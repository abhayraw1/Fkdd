import pandas as pd

class KDD:

	def __init__(self, config):
		self.data_file = config['data']
		self.feature_file = config['features']
		self.features = self.get_features()
		self.feature_names = self.get_feature_keys()
		self.feature_types = self.get_feature_types()
		self.dataframe = self.get_data()
		self.len_data = len(self.dataframe)
		self.data = self.dataframe.drop('class',1)
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
		header = self.feature_names[:]
		header.append('class')
		data = pd.read_csv(self.data_file, header=None, names=header)
		return data.sample(frac=1)
	
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

	def remove_col(self, l):
		copy_names = self.feature_names[:]
		for i in l:
			name = copy_names[i]
			print 'removing ' + name
			index = self.feature_names.index(name)
			self.feature_names.remove(name)
			self.feature_types.pop(index)
			del self.data[name]

	def get_train_data(self, train_ratio, labels=True):
		num_train_rows = int(train_ratio * self.len_data)
		self.trainX = self.data[:num_train_rows]
		self.trainY = self.labels[:num_train_rows]
		self.testX = self.data[num_train_rows:]
		self.testY = self.labels[num_train_rows:]
		return self.trainX, self.trainY

	def get_test_data(self, labels=True):
		try:
			return self.testX, self.testY
		except:
			print 'Call get_train_data first to generate train and test samples'
			return
