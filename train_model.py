import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset Titanic desde el enlace
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Preprocesamiento de los datos
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
data['Embarked'].fillna('S', inplace=True)
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

# Seleccionar las características y la variable objetivo
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de clasificación (RandomForest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Guardar el modelo entrenado en un archivo .sav
with open('model.sav', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Modelo entrenado y guardado correctamente.")
