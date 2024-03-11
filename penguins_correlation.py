# CORRELATION
# -------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
# ---------------------------------------------------------------------------------------------

penguins = pd.read_csv("penguins_size.csv", sep=',')

# Rimuovere righe con valori mancanti
penguins = penguins.dropna()

x = penguins.iloc[:,4]  # colonna 1 - flipper length
y = penguins.iloc[:,5] / 1000  # colonna 2 - body mass from g to kg
x1 = sm.add_constant(x)

# ---------------------------------------------------------------------------------------------

penguins.rename(columns={'species': 'Specie', 'island': 'Isola', 
                         'culmen_length_mm': 'Lunghezza becco (mm)', 
                         'culmen_depth_mm': 'Profondità becco (mm)', 
                         'flipper_length_mm': 'Lunghezza ali', 
                         'body_mass_g': 'Massa corporea'}, inplace=True)
penguins.drop(penguins.columns[6], axis = 1)

# ---------------------------------------------------------------------------------------------

model = sm.OLS(y, x1)  # Ordinary Least Squares
results = model.fit()

print(results.summary())
print("Parameters: ", results.params)
print("R2: ", results.rsquared)

pred = results.get_prediction()

ivlow = pred.summary_frame()["obs_ci_lower"]  # lower interval
ivup = pred.summary_frame()["obs_ci_upper"]  # upper interval

# ---------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, y, "o", label="data")
ax.plot(x, results.fittedvalues, "r--.", label="OLS - Ordinary Least Squares")
ax.plot(x, ivup, "r--")
ax.plot(x, ivlow, "r--")
ax.legend(loc="best")
plt.xlabel(penguins.columns[4] + " (mm)", fontweight='bold')
plt.ylabel(penguins.columns[5] + " (Kg)", fontweight='bold')

p = np.corrcoef(x, y)  # pearson xx, xy, yx, yy

print(f"\nNumpy: \nPearson = {p}")

r = pd.Series(x).corr(y)
rho = pd.Series(x).corr(y, method='spearman')
tau = pd.Series(x).corr(y, method='kendall')

print(f"\nPandas: \n\t- pearson: {r:.5f} \n\t\t- spearman: {rho:.5f} \n\t\t\t-kendall: {tau:.5f}\n")
plt.show()

# ---------------------------------------------------------------------------------------------