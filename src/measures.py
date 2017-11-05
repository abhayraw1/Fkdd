import numpy as np
np.set_printoptions(threshold=np.nan)

class Similarity:
	def __init__(self, data):
		self.data = data
		self.cov = data.cov().as_matrix()
		self.var = data.var().as_matrix()
		self.num_features = len(self.var)
		self.feature_names = list(self.data.columns.values)
		self.result = None
	
	'''
	this is currently returning the 
	similarity between the two given features
	'''
	def correlation(self):
		rel = np.zeros(self.cov.shape)
		for i in range(self.num_features):
			for j in range(i, self.num_features):
				rel[i,j] = self.cov[i,j]/np.sqrt(self.var[i]*self.var[j])
				rel[j,i] = rel[i,j]
		self.corr = rel
		return 1 - np.abs(rel)

	def lsre(self):
		if not hasattr(self, 'corr'):
			self.correlation()
		rel = np.zeros(self.cov.shape)
		for i in range(self.num_features):
			for j in range(self.num_features):
				rel[i,j] = self.var[j]*(1 - self.corr[i,j]**2)
		self.lsre_res = rel
		return rel

	'''
	Focusses on the  dissimilarity between the two features
	thats a guess...
	'''

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
		self.mici_res = rel
		return rel

	def compute_k_dissimilarity(self, k, red_f):
		dissimilarity = 1 - self.result
		# dissimilarity[red_f, :] = np.inf
		# dissimilarity[:, red_f] = np.inf
		k_dissimilarity = np.sort(dissimilarity)
		neighbours = np.argsort(dissimilarity)
		return k_dissimilarity[:,:k], neighbours[:,:k]

	def reduce_features(self, k, measure):
		criteria = getattr(self, measure)
		self.result = criteria()
		original_features = range(self.num_features)
		red_features = original_features[:]
		# print red_features

		while True:
		# for each feature f_i belonging to R compute r_ki

			r, neighbours = self.compute_k_dissimilarity(k, red_features)
			# print r, neighbours
			f_prime = np.argmin(r, axis=0)[k-1]
			r_k_f_prime = r[f_prime, k-1]

			# print neighbours[f_prime, :]
			# print neighbours[f_prime,:]
			for i in neighbours[f_prime, :]:
				red_features.remove(i)
			self.result = np.delete(self.result, neighbours[f_prime, :], axis=1)
			self.result = np.delete(self.result, neighbours[f_prime, :], axis=0)

			# print red_features
			
			if k > len(red_features) - 1:
				k = len(red_features) - 1
			if k == 1:
				return red_features
			# print self.result.shape
			eps = r_k_f_prime
			# print r_k_f_prime

			r, neighbours = self.compute_k_dissimilarity(k, red_features)
			f_prime = np.argmin(r, axis=0)[k-1]
			r_k_f_prime = r[f_prime, k-1]
			# print r_k_f_primes
			while r_k_f_prime > eps:
				k = k - 1
				# print k
				# print r
				f_prime = np.argmin(r, axis=0)[k-1]
				# print f_prime
				r_k_f_prime = r[f_prime, k-1]
				# print r_k_f_prime
				if k == 1:
					return red_features 
			#  for each feature compute 
			# return red_data 
