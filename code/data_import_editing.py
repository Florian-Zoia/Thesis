import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Einlesen der Iris Datensätze
df = pd.read_csv('../datensätze/iris_dirty.csv',
                 header = None # Die Datei hat keinen header deswegen werden die Namen der Felder nun mitgegeben
                 , names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# Den Head ausgeben vom vorliegenden Datensatz
print(df.head())

# Die Anzahl der einzelnen Werte pro Spalte ausgeben
print(df.count())

# Den fehlenden Wert in der Spalte "sepal width" ausgeben damit wir den Datensatz mit einem  Mittelwert befüllen können
print(df[df['sepal width'].isnull()])

# Alle Datens#tze der Irisart Iris-versicolor, um den Mittelwert zu bestimmen
iris_versicolor = df[df['class'] == 'Iris-versicolor']

# Mittelwert bestimmen aller Iris-versicolor
averageVersicolorSepalWidth = pd.Series.mean(iris_versicolor['sepal width'])
print(averageVersicolorSepalWidth)

# Den Mittelwert an der fehlenden Stelle speichern
df.loc[82, 'sepal width'] = averageVersicolorSepalWidth

# Überprüfen, ob alle Werte nun befüllt sind
print(df.count())

# Alle doppelten Datensätze ausgeben
print(df[df.duplicated(keep=False)])

# Durch die Dopplung haben wir bemerkt, dass es einen Datensatz zu oft gibt
# (Wir wissen es soll nur 150 Datensätze geben)
# Aus diesem Grund können wir nun die entsprechenden Gruppen zählen und die Duplette entfernen
print(df.groupby('class').count())
# Wir haben bemerkt, dass es eine Iris-versicolor zu viel und einen Schreibfehler bei der Iris-setosa gibt

# Wir entfernen den Datensatz mit dem Index 100, da dieser zu viel ist
df = df.drop(df.index[[100]])

# Überprüfen, ob alle Werte nun korrekt sind
print(df.count())

# Wir machen den Tippfehler ausfindig (49)
print(df[df['class'] == 'Iris-setsoa'])

# .. und setzen die Klasse neu
df.loc[49, 'class'] = 'Iris-setosa'

# Überprüfen, ob alle Werte nun korrekt sind
print(df.groupby('class').count())


# da in unserem Datensatz die petal width zum einen in Millimetern und zum anderen als String gespeichert ist
# Müssen wir dies auch auf cm vereinheitlichen und die mm entfernen
def convert_from_mm(row):
    return pd.to_numeric(row['petal width'].replace(' mm', '')) / 10


# Hier wenden wir die Funktion convert_from_mm auf jeden Wert in df['petal width'] an
df['petal width'] = df.apply(convert_from_mm, axis='columns')

# Überprüfen, ob alle Werte nun korrekt sind
print(df.head())

# umfassende, aber kompakte Darstellung statistischer Daten unseres Datensatzes
print(df.describe())

# Histogramme der obigen statistik ausgeben lassen
df.hist(figsize=(15, 15))
#plt.show() # Anzeigen der Diagramme

# Diese Histogramme lassen sich auch noch spezifizieren:
print(df.groupby('class').describe())
df.groupby('class').hist(figsize=(10, 10))
#plt.show() # Anzeigen der Diagramme

# Dies ist eine andere Darstellung der Histogramme 
sns.jointplot([df['sepal length'], df['petal length']])
#plt.show()

# Nun korrigieren wir den ausreßenden Wert auf 5.8
df.loc[143, 'sepal length'] = 5.8

# Überprüfen ob die Korrektur Erfolgreich war
sns.jointplot([df['sepal length'], df['petal length']])
#plt.show()

# Nun erzeugen wir eine Korrelationsmatrix, die einen dichten und schnellen Überblick über den Datensatz erzeugt
corrmat = df.corr()
sns.heatmap(corrmat, annot=True)
#plt.show()

# eine neue Datei speichern 
df.to_csv('../datensätze/iri_clean.csv')
