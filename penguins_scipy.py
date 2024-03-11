# SCIPY
# -------------------------------------------------------------------------------------

import pandas as pd,numpy as np, matplotlib.pyplot as plt

from colorama import Fore
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Leggere i dati e pulirli
penguins = pd.read_csv("penguins_size.csv", sep=',').dropna()

# Rinominare le colonne
penguins.rename(columns={'species': 'Specie', 'island': 'Isola', 
                         'culmen_length_mm': 'Lunghezza becco (mm)', 
                         'culmen_depth_mm': 'Profondità becco (mm)', 
                         'flipper_length_mm': 'Lunghezza ali (mm)', 
                         'body_mass_g': 'Massa corporea (kg)'}, inplace=True)

# Modificare la Massa corporea (kg) in kg
penguins['Massa corporea (kg)'] /= 1000

# Definire le variabili indipendenti (X) e dipendenti (y)
X = penguins[['Lunghezza becco (mm)', 'Profondità becco (mm)', 'Lunghezza ali (mm)', 'Massa corporea (kg)']]
y = penguins['Specie']

# Dividere i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Inizializzare e addestrare il classificatore MLP
classifier = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam', random_state=1)
classifier.fit(X_train, y_train)

# Prevedere le classi per il set di test
y_pred = classifier.predict(X_test)

# Calcolare l'accuratezza del modello e visualizzare la matrice di confusione
confmat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of MLP Classifier:", accuracy)
print(f"\n{Fore.GREEN}/// Accuracy % = {accuracy*100:.1f} %")
print("(Percettrone multi-livello)")
print(f'{Fore.WHITE}')
print("\nConfusion matrix:")
print(confmat)
# Visualizzare i grafici
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for target_name in np.unique(y):
    X_plot = X[y == target_name]
    ax1.scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], label=target_name)
    ax2.scatter(X_plot.iloc[:, 2], X_plot.iloc[:, 3], label=target_name)

ax1.set_xlabel('Lunghezza becco (mm)', fontweight='bold')
ax1.set_ylabel('Profondità becco (mm)', fontweight='bold')
ax1.legend()

ax2.set_xlabel('Lunghezza ali (mm)', fontweight='bold')
ax2.set_ylabel('Massa corporea (kg)', fontweight='bold')
ax2.legend()

plt.show()
