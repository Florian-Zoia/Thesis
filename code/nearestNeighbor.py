from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn import neighbors 

# Iris Datensatz
iris = datasets.load_iris()

# Aufteilung des Datensatzes in X(Features) und y(Labels)
X = iris.data
y= iris.target 

# Aufteilung der Features in die 4 verschiedenen Kategorien
X_sepal_length = X[:, 0] # Länge des Kelchblattes(Sepalum)
X_sepal_width = X[:, 1] # Breite des Kelchblattes(Sepalum)
X_petal_length = X[:, 2] # Länge des Kronblattes(Petalum)
X_petal_width = X[:, 3] # Breite des Kronblattes(Petalum)

# Aufteilung der Datensätze in 60% Trainingsdaten und 40% Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)

# Nearest Neighbor Classifier einbauen und immer nur den nächsten bekannten Datenpunkt betrachten (Dies ist der Estimator)
clf = neighbors.KNeighborsClassifier(1)

# Nun trainieren wir den Estimator mit der fit Methode
clf.fit(X_train, y_train)

# Hiermit geben wir aus/testen wir was unsere AI mit dem NN Algorithmus für die folgenden Werte zuordnet
# Länge des Kelchblattes = 6,3
# Breite des Kelchblattes = 2,7
# Länge des Kronblattes = 5,5
# Breite des Kronblattes = 1,5
print(clf.predict([[6.3, 2.7, 5.5, 1.5]]))

# Hier wird gemessen wie gut ein Satz von Features auf einen Satz von Labels passt 0(überhaupt nicht) 1(passt perfekt)
print(clf.score(X_train, y_train))

# Dieser Wert gibt aus wie viel Prozent der Testdaten richtig vorhergesagt werden können
print(clf.score(X_test, y_test))
