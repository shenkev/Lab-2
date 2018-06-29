# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pointbiserialr, spearmanr
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


data = pd.read_csv("./data/adult.csv")
data.head()


col_names = data.columns
num_data = data.shape[0]
for c in col_names:
    num_non = data[c].isin(["?"]).sum()
    if num_non > 0:
        print (c)
        print (num_non)
        print ("{0:.2f}%".format(float(num_non) / num_data * 100))
        print ("\n")

data = data[data["workclass"] != "?"]
data = data[data["occupation"] != "?"]
data = data[data["native-country"] != "?"]

print(data.shape)

# descriptive stats for numerical fields
data.describe()

# frequency for categorical fields
category_col =['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'gender', 'native-country', 'income']
for c in category_col:
    print (c)
    print (data[c].value_counts())

print(data["income"].value_counts()[0] / data.shape[0])
print(data["income"].value_counts()[1] / data.shape[0])

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data.head())

labelencoder = LabelEncoder()

for col in category_col:
    data[col] = labelencoder.fit_transform(data[col])

# balance the dataset
# zero_data = data.loc[data['income'] == 0]
# one_data = data.loc[data['income'] == 1]
# drop_indices = np.random.choice(zero_data.index, zero_data.shape[0] - one_data.shape[0], replace=False)
# zero_data = zero_data.drop(drop_indices)
# data = zero_data.append(one_data)

onehotencoder = OneHotEncoder()

y = data['income'].values
data = data.drop(columns=['income'])
category_col.remove('income')

cat = data[category_col]
enc = OneHotEncoder()
enc.fit(cat)
print(enc.n_values)
cat_onehot = enc.transform(cat).toarray()
cts = data.drop(columns=category_col)
cts=(cts-cts.mean())/cts.std()
cts_np = cts.values
final_data = np.concatenate([cts_np, cat_onehot], axis=1)

from sklearn.model_selection import train_test_split
seed = 0
X_train, X_test, y_train, y_test = train_test_split(final_data, y, test_size=0.2, random_state=seed)


from time import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader

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
        self.nonlin1 = nn.Tanh()
        # self.fc2 = nn.Linear(128, 64)
        # self.nonlin2 = nn.PReLU()
        self.fc3 = nn.Linear(64, 1)
        self.out = nn.Sigmoid()

    def forward(self, features):

        x = self.nonlin1(self.fc1(features))
        # x = self.nonlin2(self.fc2(x))
        x = self.fc3(x)

        return self.out(x)

train_dataset = MushroomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=45000, num_workers=1, shuffle=True)
test_dataset = MushroomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=1, shuffle=False)

loss_fnc = torch.nn.BCELoss()
model = test_network(X_train.shape[1])
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1.1)

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

    tot_corr = 0

    # decay_factor = 0.95 ** epoch
    # current_lr = 0.0001 * decay_factor
    # for group in optimizer.param_groups:
    #     group['lr'] = current_lr

    for i, sample in enumerate(train_loader):
        feats, label = sample
        feats, label = Variable(feats).float(), Variable(label)
        # batch_loss = grad_step(feats, label, model, optimizer)

        optimizer.zero_grad()

        predictions = model(feats)
        batch_loss = loss_fnc(input=predictions.squeeze(), target=label.float())
        batch_loss.backward()
        optimizer.step()

        corr = (predictions > 0.5).squeeze().long() == label
        tot_corr += int(corr.sum())

        print_loss += batch_loss

        if (step+1) % 10 == 0:
            test_corr = test(model)
            print("Epoch: {}, Step {} | Loss: {} | Time: {} | Test acc: {}".format(epoch+1, step+1,
                                                                      print_loss / 100, time() - tic, test_corr))
            print_loss, tic = 0, time()

        step = step + 1

    print("Epoch train corr: {}".format(float(tot_corr)/len(train_dataset)))
