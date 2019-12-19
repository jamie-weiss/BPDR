from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import decimal
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn
import warnings



def feature_importance(x, y):
	clf = RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=10)
	model = clf.fit(x, y)
	importances = model.feature_importances_
	sort = np.argsort(importances)
	feature_importance_indecies = sort[::-1]
	return feature_importance_indecies


def normalize_data(data):
	scaler = MinMaxScaler()
	return scaler.fit_transform(data)


def scale_data(data, scaler):
	scaled = []
	for row in data:
		scaled_row = []
		for val in row:
			scaled_row.append(int(round(val * scaler)))
		scaled.append(scaled_row)
	return scaled


def data_to_binary(data, bpcps):
	binary = []
	for row in data:
		bin_row = []
		for value in row:
			bin_row.append(format(value, '0' + str(int(bpcps)) + 'b'))
		binary.append(bin_row)
	return binary


def chunks(l, n):
	for i in range(0, len(l), n):
		yield l[i:i + n]


def pack_bits(binary, feature_importance_indecies, cols_per_string):
	c = chunks(feature_importance_indecies, int(cols_per_string))
	c_list = list(c)
	packed = []
	for row in binary:
		new_row = []
		for indecies in c_list:
			bit_string = ''
			for i in indecies:
				bit_string += row[i]
			new_row.append(bit_string)
		packed.append(new_row)
	return packed     


def to_int(packed_binary):
	packed_integers = []
	for row in packed_binary:
		packed_row = []
		for value in row:
			packed_row.append(int(value,2))
		packed_integers.append(packed_row)
	return packed_integers


def unscale(packed_integers, scale):
	unscaled = []
	for row in packed_integers:
		unscaled_row = []
		for value in row:
			unscaled_row.append(float(value / scale))
		unscaled.append(unscaled_row)
	return unscaled


def drop_unimportant_columns(data, feature_importance_indecies, cols_to_drop):
	i_to_drop = []
	for i in range(0, cols_to_drop):
		i_to_drop.append(feature_importance_indecies[-1-i])
	dropped_data = []
	for row in data:
		dropped_row = row.copy()
		for j in i_to_drop:
			dropped_row = np.delete(dropped_row, j)
		dropped_data.append(dropped_row)
	return dropped_data


def get_possible_values(l):
	list_set = set(l)  
	unique_list = (list(list_set))
	unique_list.sort()
	return unique_list


def split_data_on_label(data, targets):
	possible_labels = get_possible_values(targets)
	label_split = {}
	for l in possible_labels:
		label_split[l] = []
	for i in range(len(data)):
		label_split[targets[i]].append(data[i])
	return label_split     





class BPDR:

	def __init__(self, n_components=None):
		self.n_components = n_components

	
	def fit_transform(self, X, y=None):
		
		n_samples, n_features = np.shape(X)
		if not 1 <= self.n_components <= min(n_samples, n_features):
			raise ValueError("n_components=%r must be between 1 and "
                             "min(n_samples, n_features)=%r" 
                             % (self.n_components, min(n_samples, n_features)))

		U = self._fit(X, y)
		return U

	
	def _fit(self, data, targets):
		num_columns = np.shape(data)[1]
		if num_columns / self.n_components > 15:
			warnings.filterwarnings("error")
			try:
				clf = LinearDiscriminantAnalysis(n_components=30)
				data = clf.fit_transform(data, targets)
			except:
				clf = PCA(n_components=30)
				data = clf.fit_transform(data)
		num_columns = np.shape(data)[1]
		cols_per_slice = math.floor(num_columns / self.n_components)
		# If doesnt divide evenly, we drop unimportant columns (to be updated in future versions)
		cols_to_drop = num_columns % self.n_components
		if targets is not None:
			feature_importance_indecies = feature_importance(data, targets)
		else:
			feature_importance_indecies = []
			for i in range(num_columns):
				feature_importance_indecies.append(i)
			feature_importance_indecies = np.array(feature_importance_indecies)   
		if cols_to_drop != 0:
			data = drop_unimportant_columns(data, feature_importance_indecies, cols_to_drop)
			feature_importance_indecies = []
			for i in range(len(data[0])):
				feature_importance_indecies.append(i)
			feature_importance_indecies = np.array(feature_importance_indecies)

		normalized = normalize_data(data)

		cols_per_string = math.floor(num_columns / self.n_components)
		bpcps = 128 / cols_per_string
		num = (2 ** bpcps) - 1
		digits = len(str(int(num)))
		exp = digits - 1

		scale = (10 ** exp)

		norm_scaled = scale_data(normalized, scale)
		binary = data_to_binary(norm_scaled, bpcps)

		packed_binary = pack_bits(binary, feature_importance_indecies, cols_per_string)
		packed_integers = to_int(packed_binary)
		unscaled = unscale(packed_integers, scale)

		if targets is not None:
			split_dict = split_data_on_label(unscaled, targets)
			length = len(split_dict.keys())
			variances = []
			for key in split_dict.keys():
				arr = split_dict[key]
				var = np.var(arr)
				variances.append(var)
			var_mean = sum(variances) / float(length)
			scaled_vars = []
			for v in variances:
				scaled_vars.append(v/var_mean)

			self.variances = variances
			self.scaled_variances = scaled_vars
			self.mean_variance = var_mean
		else:
			self.variances = None
			self.scaled_variances = None
			self.mean_variance = None
			
		return unscaled

