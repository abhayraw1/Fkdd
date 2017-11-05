# Fkdd
Fast Feature Reduction Technique for Classification of KDD-99 dataset

## The Dataset
The dataset can be found [here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

## What is this repository, you ask?
Well, this is a simple implementation of the KDD-dataset classification using feature reduction techniques.
The KDD datasets has 41 types of different features which when looked upon closely may or may not contibute towards the classification of the data (as detected intrusion or otherwise). This technique follows the removal of such features which do not contribute(or have a very litte contibution) towards the classification of the data.
Some of these features may be highly correlated as we may see. And keeping them is just going to make our CPU's job a lost harder.
