

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_learning_curves
from sklearn.model_selection import train_test_split


# carrega o conjunto de treinamento e teste
train = pd.read_csv('dataset/Titanic/train.csv')
test = pd.read_csv('dataset/Titanic/test.csv')

print(test.head())

# deleta as informações irrelevantes
train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# transforma as colunas de palavras em numeros
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

# ver valores nulos
#new_data_train.isnull().sum().sort_values(ascending=False).head(10)

# aplica uma correção para os valores nulos - setar  média entre eles
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)

# define o conjunto de features e labels
X = new_data_train.drop('Survived', axis=1)
y = new_data_train['Survived']

# separa os dados em um conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(y_test)

#cria o modelo
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train,y_train)

plot_learning_curves(X_train, y_train, X_test, y_test, clf=tree)
plt.title("DecisionTreeClassifier %.2f " %(float(tree.score(X_test,y_test))))
plt.show()

#verificando o Score do conjunto de treino
#print('Score : ', tree.score(X,y))
#print('Score : ', clf_rf.score(X_test,y_test))


submission = pd.DataFrame()
submission['PassengerId'] = X_test['PassengerId']
submission['Survived'] = tree.predict(X_test)
submission.to_csv('prediction/Titanic/submission.csv', index=False)