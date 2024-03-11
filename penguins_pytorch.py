# PYTORCH
# -------------------------------------------------------------------------------------

from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from colorama import Fore

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, input_features=4, hidden_layer1=25, hidden_layer2=30, output_features=3):
        super().__init__()
        self.input = nn.Linear(input_features, hidden_layer1)
        self.hidden = nn.Linear(hidden_layer1, hidden_layer2)
        self.output = nn.Linear(hidden_layer2, output_features)
    
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)
        return x

# ----------------------------------------------------------------------------------------------------

penguins = pd.read_csv("penguins_size.csv", sep=',').dropna()

X = penguins.iloc[:, 2:6]
y = penguins['species']

names = penguins.columns[2:6]
feature_names = penguins.columns[1]
targets = np.unique(y)

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

test_size = 0.2
indices = np.random.permutation(len(penguins))
n_test_samples = int(test_size * len(penguins))  # 0.2 * 150

X_train = X_scaled[indices[:-n_test_samples]]
X_test = X_scaled[indices[-n_test_samples:]]

le = preprocessing.LabelEncoder()

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Corrected lines:
y_train = y.iloc[indices[:-n_test_samples]]
y_test = y.iloc[indices[-n_test_samples:]]

y_train = torch.LongTensor(le.fit_transform(y_train))
y_test = torch.LongTensor(le.fit_transform(y_test))

# ----------------------------------------------------------------------------------------------------

model = Model()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100 # quante volte si vuole che il modello veda il train dataset
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = loss_fn(y_pred, y_train)
    losses.append(loss.item())
    print(f'Epoch: {(i+1):2}/{epochs} - Loss: {loss.item():.3f}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

la = np.array(losses)

plt.figure()
plt.plot(range(epochs), la)
plt.ylabel('Perdite', fontweight='bold')
plt.xlabel('Epoch', fontweight='bold')
plt.title('Perdita dati', fontweight='bold')
plt.show()

# ----------------------------------------------------------------------------------------------------

enc = OneHotEncoder()

Y_onehot = enc.fit_transform(y_test[:, np.newaxis]).toarray()

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_prob = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    
    # Convertire le predizioni in array numpy
    y_pred_np = y_pred.numpy()

    # Convertire le etichette di test in array numpy
    y_test_np = y_test.numpy()

    fpr, tpr, thresholds = roc_curve(Y_onehot.ravel(), y_pred_prob.numpy().ravel())

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc(fpr, tpr)))
plt.xlabel('Ratio Falsi Positivi', fontweight='bold')
plt.ylabel('Ratio Veri Positivi', fontweight='bold')
plt.title('Curva ROC',  fontweight='bold')
plt.legend()
plt.show()

accuracy = accuracy_score(y_test_np, y_pred_np)
print(f'\n/// {Fore.GREEN}Accuratezza del MLP: {((accuracy)*100):.2f}%\n')