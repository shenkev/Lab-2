import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.autograd import Variable


mushrooms = pd.read_csv('./data/mushrooms.csv')

labelencoder = LabelEncoder()
for col in mushrooms.columns:
    mushrooms[col] = labelencoder.fit_transform(mushrooms[col])

mushrooms.head()
mushrooms['stalk-color-above-ring'].unique()
print(mushrooms.groupby('class').size())

X = mushrooms.iloc[:,1:23]  # all rows, all the features and no labels
y = mushrooms.iloc[:, 0]

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(X.describe())


X = X.drop(columns=['veil-type', 'odor', 'population', 'ring-type', 'gill-color'])

enc = OneHotEncoder()
enc.fit(X)
enc.n_values
X_onehot = enc.transform(X).toarray()
y = y.values

from sklearn.model_selection import train_test_split
seed = 0
X_train, X_test, y_train, y_test = train_test_split(X_onehot, y, test_size=0.2, random_state=seed)


class MushroomDataset(data.Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        categorical = self.X[index]
        label =self.y[index]

        return categorical, label

class test_network(nn.Module):

    def __init__(self, input_size):

        super(test_network, self).__init__()

        self.input_size = input_size
        self.output_size = 1

        self.fc1 = nn.Linear(self.input_size, 64)
        self.nonlin1 = nn.PReLU()
        self.fc2 = nn.Linear(64, 1)
        self.out = nn.Sigmoid()

    def forward(self, features):

        x = self.nonlin1(self.fc1(features))
        x = self.fc2(x)

        return self.out(x)

train_dataset = MushroomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, num_workers=1, shuffle=True)
test_dataset = MushroomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=1, shuffle=False)

loss_fnc = torch.nn.BCELoss()
model = test_network(X_train.shape[1])
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)


def grad_step(feats, labels, model, optim):
    optim.zero_grad()

    predictions = model(feats)
    loss = loss_fnc(input=predictions.squeeze(), target=labels.float())
    loss.backward()
    optim.step()

    return loss

def test(model):
    total_corr = 0

    for i, sample in enumerate(test_loader):
        feats, label = sample
        feats, label = Variable(feats).float(), Variable(label)
        prediction = model(feats)
        corr = (prediction > 0.5).squeeze().long() == label
        total_corr += int(corr.sum())

    return float(total_corr)/len(test_loader.dataset)

step = 0
for epoch in range(100):
    print_loss, tic = 0, time()

    for i, sample in enumerate(train_loader):
        feats, label = sample
        feats, label = Variable(feats).float(), Variable(label)
        batch_loss = grad_step(feats, label, model, optimizer)

        print_loss += batch_loss

        if (step+1) % 10 == 0:
            test_corr = test(model)
            print("Epoch: {}, Step {} | Loss: {} | Time: {} | Test acc: {}".format(epoch+1, step+1,
                                                                      print_loss / 100, time() - tic, test_corr))
            print_loss, tic = 0, time()

        step = step + 1
