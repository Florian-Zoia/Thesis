import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

# Den Boston Datensatz, der die Preise + 13 Attribute von Bostoner Vorstädten enthält importieren 
boston = pd.read_csv('../datensätze/boston_housing.csv')
boston_np = boston.to_numpy()

print(boston)

# überprüfen ob der Datensatz stimmt 
print(boston.head())

#Anzahl innerhalb jeder Spalte 
print(boston.count())
#Anzahl innerhalb der Zeile 
print(boston.shape[1])

# wir speichern Features und Labels X(Features) und y(Labels)
print(np.shape(boston))

X = boston_np[:, 5:6]
y = boston_np[:, 13]

# Eine Darstellung des Features und des Vektors 
a = 15.; b = -70.
lx = np.arange(4,10)
lguess = a*lx + b

plt.plot(lx, lguess, c='red')
plt.scatter(X, y, marker='.', alpha=0.5, color='blue')
plt.xlim(0, 10) 
plt.ylim(0, 60)
plt.xlabel('Durchschnittl. Anzahl Räume')
plt.ylabel('Preis ($1000)')
#plt.show()

# Wir erstellen eine Instanz welche die Lineare Regression anwendet und trainieren sie mit der fit Methode 
lr = lm.LinearRegression() # Regressorinstanz
lr.fit(X, y)

# Wir wir die Güte des Modells ausgegeben je näher an 1 desto besser 
print(lr.score(X, y))

# Hier werden die Preise predicted aus den Trainingsdaten 
y_pred = lr.predict(X)

# Hier erstellen wir eine Darstellung der Werte
# Dicke rote Linie AI predicted prices 
plt.scatter(X, y, marker='.', alpha=0.5, color='blue')
plt.scatter(X, y_pred, marker='.', alpha=0.5, color='red')
plt.xlim(0, 10)
plt.ylim(0, 60)
plt.xlabel('Durchschnittl. Anzahl Räume')
plt.ylabel('Preis ($1000)')

plt.show()
