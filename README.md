# Titanic Modelo de Predicción de Supervivencia

Este proyecto utiliza un modelo de clasificación basado en Random Forest para predecir la probabilidad de supervivencia de los pasajeros del Titanic.

## Estructura del Proyecto

El proyecto está dividido en tres partes principales:
1. **Modelo de Clasificación**: Un modelo entrenado con datos del Titanic para predecir si un pasajero sobrevivió o no.
2. **API FastAPI**: Un servidor FastAPI que permite a los usuarios enviar datos a través de un formulario o API para realizar predicciones de supervivencia.
3. **Interfaz HTML**: Una página HTML que permite al usuario interactuar con el modelo de predicción a través de un formulario web.

## Instalación

1. Clona el repositorio en tu máquina local:
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    ```

2. Instala los paquetes necesarios:
    ```bash
    pip install -r requirements.txt
    ```

3. Ejecuta la aplicación FastAPI:
    ```bash
    uvicorn app:app --reload
    ```

4. Accede a la interfaz web abriendo tu navegador y visitando `http://localhost:8000`.

## Uso

- **Predicción desde el Formulario Web**: Rellena los campos del formulario en la página principal y presiona "Predecir" para obtener la predicción sobre la supervivencia del pasajero.
- **Predicción desde la API**: Envía una solicitud POST a `/predict` con los datos del pasajero en formato JSON.
- **Predicción desde un archivo CSV**: Envía una solicitud GET a `/predict_from_csv?url=<URL_DEL_CSV>` para obtener predicciones desde un archivo CSV con datos de pasajeros.

## Tecnologías Usadas

- **Python**: Para la creación del modelo de predicción.
- **FastAPI**: Para crear la API que maneja las solicitudes y respuestas.
- **HTML/CSS**: Para la interfaz de usuario.
- **Scikit-learn**: Para el modelo de Machine Learning (Random Forest).
- **Pandas**: Para el manejo de los datos.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
