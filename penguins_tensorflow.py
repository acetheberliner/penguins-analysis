# TENSORFLOW
# -------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn import preprocessing

from colorama import Fore
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
# -------------------------------------------------------------------------------------

penguins = pd.read_csv('penguins_size.csv', sep=',').dropna()

X = penguins.iloc[:, 2:6]
y = penguins['species']

# categorical to numeric
le = preprocessing.LabelEncoder()
y1 = le.fit_transform(y)

penguins["species"] = penguins["species"].map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})
Y = pd.get_dummies(y1).values
# -------------------------------------------------------------------------------------

# Scale data to have 0 means and variance 1, it helps convergence
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
test_size = 0.2 # set 20% of dataset as testing material
indices = np.random.permutation(len(penguins))
n_test_samples = int(test_size * len(penguins))

X_train = X_scaled[indices[:-n_test_samples]]
y_train = Y[indices[:-n_test_samples]]

X_test = X_scaled[indices[-n_test_samples:]]
y_test = Y[indices[-n_test_samples:]]

# -------------------------------------------------------------------------------------
# MLP topology
nhid1 = 4 # hidden layer 1
nhid2 = 4 # hidden layer 2
nout = 3 #output neurons

model = tf.keras.Sequential([
    tf.keras.layers.Dense(nhid1,input_dim=4, activation='relu'), # 1st hidden
    tf.keras.layers.Dense(nhid2,activation='relu'), # 2nd hidden
    tf.keras.layers.Dense(nout, activation='softmax') # out nuerons
], name='MLP')

print(model.summary()) # <-----

optimizer = Adam(learning_rate=0.01)
loss = tf.keras.losses.categorical_crossentropy

#tf.keras.losses.mean_squared_error
model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

history = model.fit(X_train, y_train, batch_size=5, epochs=100,verbose=1) # the model will "see" the dataset 100 times (epochs)
loss_train, train_accuracy = model.evaluate(X_train, y_train)
loss_test, test_accuracy = model.evaluate(X_test, y_test)

print(f'\nThe training set accuracy for the model is {train_accuracy}\n The test set accuracy for the model is {test_accuracy}\n')
print(f'/// {Fore.GREEN} MLP test set accuracy {(test_accuracy*100):.2f}%///\n')

y_pred = model.predict(X_test)
actual = np.argmax(y_test, axis=1)
predicted = np.argmax(y_pred, axis=1)

print(f"Actual: {actual}")
print(f"Predicted: {predicted}")
hist_val = model.fit(X_test, y_test, batch_size=5, epochs=100,verbose=1)
# -------------------------------------------------------------------------------------

# Hereplot the training and validation loss and accuracy
fig, ax = plt.subplots(1,2,figsize = (12,4))

ax[0].plot(history.history['loss'], 'r',label = 'Training Loss')
ax[0].plot(hist_val.history['loss'],'b',label = 'Validation Loss')
ax[1].plot(history.history['accuracy'], 'r',label = 'Training Accuracy')
ax[1].plot(hist_val.history['accuracy'],'b',label = 'Validation Accuracy')

ax[0].legend()
ax[1].legend()

ax[0].set_xlabel('Epochs')
ax[1].set_xlabel('Epochs');
ax[0].set_ylabel('Loss')
ax[1].set_ylabel('Accuracy %');

fig.suptitle('MLP Training', fontsize = 24)

plt.show()
# --------------------------------------------------------------------------------------

# Receiver Operating Characteristic (ROC)
Y_pred = model.predict(X_test)

plt.figure(figsize=(7, 5))

plt.plot([0, 1], [0, 1], 'k--')
fpr, tpr, threshold = roc_curve(y_test.ravel(), Y_pred.ravel())
plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(model, auc(fpr, tpr)))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

plt.title('ROC curve')
plt.legend();

plt.show()
# --------------------------------------------------------------------------------------