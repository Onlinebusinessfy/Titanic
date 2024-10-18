import pickle

# Función para cargar el modelo desde un archivo .sav
def load_model(path='model.sav'):
    with open(path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model
