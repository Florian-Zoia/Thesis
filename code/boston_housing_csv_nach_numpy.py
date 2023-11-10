import pandas as pd
import numpy as np

# Den Boston Datensatz, der die Preise + 13 Attribute von Bostoner Vorstädten enthält importieren 
df = pd.read_csv('../datensätze/boston_housing.csv', header = 0)

# überprüfen ob der Datensatz stimmt 
#print(boston.head())

#Anzahl innerhalb jeder Spalte 
#print(boston.count())
#Anzahl innerhalb der Zeile 
#print(boston.shape[1])

# Wir speichern die csv Datei in ein numpy Array damit wir es wie in sklearn nutzen können 
# put the original column names in a python list
original_headers = list(df.columns.values)

# remove the non-numeric columns
df = df._get_numeric_data()

# put the numeric column names in a python list
numeric_headers = list(df.columns.values)

# create a numpy array with the numeric values for input into scikit-learn
numpy_array = df.to_numpy()

# reverse the order of the columns
numeric_headers.reverse()
reverse_df = df[numeric_headers]

# write the reverse_df to an excel spreadsheet
reverse_df.to_excel('path_to_file.xls')


# wir speichern Features und Labels X(Features) und y(Labels)
