import numpy as np
np.set_printoptions(threshold=np.nan)

class Similarity:
	def __init__(self, data):
		self.data = data
		self.cov = data.cov().as_matrix()
		self.var = data.var().as_matrix()
		self.num_features = len(self.var)
		self.corr = None
		self.feature_names = list(self.data.columns.values)
	
	def correlation(self):
		rel = np.zeros(self.cov.shape)
		for i in range(self.num_features):
			for j in range(i, self.num_features):
				rel[i,j] = self.cov[i,j]/np.sqrt(self.var[i]*self.var[j])
				rel[j,i] = rel[i,j]
		self.corr = rel
		return rel

	def lsre(self):
		if not self.corr:
			self.correlation()
		rel = np.zeros(self.cov.shape)
		for i in range(self.num_features):
			for j in range(self.num_features):
				rel[i,j] = self.var[j]*(1 - self.corr[i,j]**2)
		return rel

	def mici (self):
		rel = np.zeros(self.cov.shape)
		for i in range(self.num_features):
			for j in range(i, self.num_features):
				cov_ij = self.cov[np.ix_([i,j], [i,j])]
				try:
					eig_vals, eig_vecs = np.linalg.eig(cov_ij)
				except:
					eig_vals = [np.inf]
				rel[i,j] = min(eig_vals)
				rel[j,i] = rel[i,j]
		return rel
