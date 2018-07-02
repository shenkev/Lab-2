import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from util import *
from model import MultiLayerPerceptron
from dataset import AdultDataset


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html
"""
seed = 0

###############
# LOAD DATASET    #
###############
data = pd.read_csv("./data/adult.csv")

###############
# DATA VISUALIZATION    #
###############

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

# we can check the number of rows and columns in the dataset using the .shape field
print(data.shape)

# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
verbose_print(data.head())

# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# let's begin by checking how balanced our dataset is. We can use the .value_counts() method to check how many entries
# in the dataset have a particular value for a field.
print(data["income"].value_counts()[0] / data.shape[0])
print(data["income"].value_counts()[1] / data.shape[0])

###############
# DATA CLEANING    #
###############

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
for feature in col_names:
    num_missing = data[feature].isin(["?"]).sum()
    if num_missing > 0:
        print(feature)
        print(num_missing)
        print("Missing percent: {0:.2f}% \n".format(float(num_missing) / num_rows * 100))

# let's remove all the rows with missing entries
data = data[data["workclass"] != "?"]
data = data[data["occupation"] != "?"]
data = data[data["native-country"] != "?"]

# finally let's check how many rows remain in our dataset
print(data.shape)

###############
# DATA STATISTICS    #
###############

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method
verbose_print(data.describe())

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats =['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
for feature in categorical_feats:
    print(feature)
    print(data[feature].value_counts())

# visualize the first 3 features
for i in range(3):
    binary_bar_chart(data, categorical_feats[i])
    pie_chart(data, categorical_feats[i])

###############
# DATASET BALANCING    #
###############

# balance the dataset
# zero_data = data.loc[data['income'] == 0]
# one_data = data.loc[data['income'] == 1]
# drop_indices = np.random.choice(zero_data.index, zero_data.shape[0] - one_data.shape[0], replace=False)
# zero_data = zero_data.drop(drop_indices)
# data = zero_data.append(one_data)

###############
# ENCODING CATEGORICAL FEATURES    #
###############

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
labelencoder = LabelEncoder()
for feature in categorical_feats:
    data[feature] = labelencoder.fit_transform(data[feature])

# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field
y = data['income']
data = data.drop(columns=['income'])
categorical_feats.remove('income')
y = y.values  # convert DataFrame to numpy array

oneh_encoder = OneHotEncoder()
cat_cols = data[categorical_feats]

oneh_encoder.fit(cat_cols)
cat_onehot = oneh_encoder.transform(cat_cols).toarray()

# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
cts_data = data.drop(columns=categorical_feats)
cts_data = (cts_data-cts_data.mean())/cts_data.std()

cts_data = cts_data.values  # convert DataFrame to numpy array

# stitch continuous and cat features
X = np.concatenate([cts_data, cat_onehot], axis=1)

###############
# MAKE THE TRAIN AND TEST SPLIT    #
###############

# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

###############
# LOAD DATA AND MODEL    #
###############

def load_data(batch_size, lr):
    train_dataset = AdultDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    test_dataset = AdultDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    loss_fnc = torch.nn.BCELoss()
    model = MultiLayerPerceptron(X_train.shape[1])
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    return train_loader, test_loader, model, loss_fnc, optimizer


def evaluate(model, test_loader):
    total_corr = 0

    for i, sample in enumerate(test_loader):
        feats, label = sample
        feats, label = Variable(feats).float(), Variable(label)
        prediction = model(feats)
        corr = (prediction > 0.5).squeeze().long() == label
        total_corr += int(corr.sum())

    return float(total_corr)/len(test_loader.dataset)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    train_loader, test_loader, model, loss_fnc, optimizer = load_data(args.batch_size, args.lr)

    t = 0

    for epoch in range(args.epochs):
        print_loss, tic = 0, time()

        tot_corr = 0

        # decay_factor = 0.95 ** epoch
        # current_lr = 0.0001 * decay_factor
        # for group in optimizer.param_groups:
        #     group['lr'] = current_lr

        for i, sample in enumerate(train_loader):
            feats, label = sample
            feats, label = Variable(feats).float(), Variable(label)

            # gradient step
            optimizer.zero_grad()

            predictions = model(feats)
            batch_loss = loss_fnc(input=predictions.squeeze(), target=label.float())
            batch_loss.backward()
            optimizer.step()

            # evaluate number of correct predictions
            corr = (predictions > 0.5).squeeze().long() == label
            tot_corr += int(corr.sum())

            print_loss += batch_loss

            # evaluate the model on the test set every eval_every steps
            if (t+1) % args.eval_every == 0:
                test_corr = evaluate(model, test_loader)
                print("Epoch: {}, Step {} | Loss: {} | Time: {} | Test acc: {}".format(epoch+1, t+1,
                                                                          print_loss / 100, time() - tic, test_corr))
                print_loss, tic = 0, time()

            t = t + 1

        print("Train acc: {}".format(float(tot_corr)/len(train_loader.dataset)))


if __name__ == "__main__":
    main()