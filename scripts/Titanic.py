
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# carrega o conjunto de treinamento e teste
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

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

#cria o modelo
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X,y)

#verificando o Score do conjunto de treino
print('Score : ', tree.score(X,y))

#submission = pd.DataFrame()
#submission['PassengerId'] = new_data_test['PassengerId']
#submission['Survived'] = tree.predict(new_data_test)

#submission.to_csv('prediction/submission.csv', index=False)