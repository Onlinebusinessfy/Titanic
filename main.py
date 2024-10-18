from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from fastapi.responses import FileResponse
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo entrenado
with open('model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

app = FastAPI()

# Estructura de los datos que recibiremos a través de la API
class TitanicData(BaseModel):
    Pclass: int
    Sex: int  # 0 para mujer, 1 para hombre
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int  # 0 = Cherbourg, 1 = Queenstown, 2 = Southampton

# Endpoint para predecir desde datos enviados por el usuario
@app.post("/predict")
async def predict_survival(data: TitanicData):
    # Convertir los datos en un DataFrame
    input_data = pd.DataFrame([data.dict().values()], columns=data.dict().keys())
    
    # Realizar la predicción
    prediction = model.predict(input_data)[0]
    
    # Devolver el resultado de la predicción
    return {"Survived": bool(prediction)}

# Ruta para servir el archivo HTML
@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

# Preprocesar y hacer predicciones desde CSV externo
@app.get("/predict_from_csv")
async def predict_from_csv(url: str):
    # Cargar el archivo CSV
    data = pd.read_csv(url)
    
    # Preprocesamiento
    data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
    data['Embarked'].fillna('S', inplace=True)
    data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    
    # Seleccionar las características
    X_new = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    
    # Hacer las predicciones
    predictions = model.predict(X_new)
    
    # Devolver las predicciones
    return {"predictions": predictions.tolist()}
