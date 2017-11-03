from KDD_wrapper import KDD
from measures import Similarity

d = {'data':'../Datasets/kdd10', 'features':'../Datasets/features'}
a = KDD(d)
s = Similarity(a.get_normalized_data())
# print a.dataframe[a.feature_names[20]].value_counts()
# print a.feature_names[20]
# print s.cov[0,19], s.cov[19,0]
# print s.correlation()
print s.mici()
# s = a.dataframe.select_dtypes(include=['object'])
# print list(s.columns)
# a.categorical_to_cont()
# print a.feature_names[0]
# print a.dataframe[a.feature_names[0]].value_counts()
# print a.data.dtypes
# print a.get_normalized_data().head()
# print a.get_standardized_data().head()
