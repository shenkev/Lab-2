import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Problem.model import MultiLayerPerceptron
from Problem.dataset import AdultDataset
from util import *


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

# =================================== LOAD DATASET =========================================== #

######

# 2.1 YOUR CODE HERE

######

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######

# 2.2 YOUR CODE HERE

######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
for feature in col_names:
    ######

    # 2.3 YOUR CODE HERE

    ######

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

# 2.4 YOUR CODE HERE

######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats =['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    ######

    # 2.4 YOUR CODE HERE

    ######

# visualize the first 3 features using pie and bar graphs

######

# 2.4 YOUR CODE HERE

######

# =================================== ENCODING CATEGORICAL FEATURES =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

######

# 2.5 YOUR CODE HERE

######
cat_cols = data[categorical_feats]

oneh_encoder.fit(cat_cols)
cat_onehot = oneh_encoder.transform(cat_cols).toarray()  # .toarray() converts the DataFrame to a numpy array


# =================================== MAKE THE TRAIN AND TEST SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

# 2.6 YOUR CODE HERE

######

# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):
    ######

    # 3.2 YOUR CODE HERE

    ######


    return train_loader, test_loader


def load_model(lr):

    ######

    # 3.4 YOUR CODE HERE

    ######

    return model, loss_fnc, optimizer


def evaluate(model, test_loader):
    total_corr = 0

    ######

    # 3.6 YOUR CODE HERE

    ######

    return float(total_corr)/len(test_loader.dataset)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    ######

    # 3.5 YOUR CODE HERE

    ######


if __name__ == "__main__":
    main()