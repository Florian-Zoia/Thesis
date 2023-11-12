import sklearn.linear_model as lm
from sklearn import datasets 

#Logistic Regression Instanz initialisieren 
logr = lm.LogisticRegression()

#Iris Datensatz laden 
iris = datasets.load_iris()

#Datensatz in Features und Klassen aufteilen 
X = iris.data
y = iris.target

print(X.shape)
print(y.shape)

#Im folgenden wird der Irisdatensatz mit der logistischen Regression ausgewertet und der Score, welcher zeigt wie richtig die KI liegen kann ausgegeben 
logr.fit(X,y)
print(logr.score(X,y))


