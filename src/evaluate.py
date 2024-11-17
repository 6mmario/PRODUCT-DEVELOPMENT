# src/evaluate.py

import sys
import json
import yaml
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(input_file, model_file, metrics_file, params_file):
    """
    Evalúa un modelo entrenado en un conjunto de datos de prueba y guarda las métricas de evaluación.
    
    Args:
        input_file (str): Ruta al archivo CSV con los datos de prueba.
        model_file (str): Ruta al archivo del modelo entrenado (formato .joblib).
        metrics_file (str): Ruta para guardar las métricas calculadas en formato JSON.
        params_file (str): Ruta al archivo YAML con la configuración de características y objetivo.
    """
    # Cargar el dataset de prueba
    df = pd.read_csv(input_file)
    print(f"Columnas en el archivo de prueba: {df.columns.tolist()}")  # Confirmar columnas disponibles

    # Leer los parámetros de configuración
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    # Validar que las columnas requeridas existan en el dataset
    missing_columns = [col for col in features + [target] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Las siguientes columnas no están en el archivo de prueba: {missing_columns}")

    # Dividir las características y la variable objetivo
    X = df[features]
    y = df[target]

    # Cargar el modelo entrenado
    model = joblib.load(model_file)
    print(f"Modelo cargado desde: {model_file}")

    # Realizar predicciones
    predictions = model.predict(X)

    # Calcular métricas de evaluación
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    # Guardar las métricas en un archivo JSON
    metrics = {
        'mse': mse,
        'r2': r2
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas guardadas en: {metrics_file}")

    # Mostrar métricas en la consola
    print(f"Error cuadrático medio (MSE): {mse}")
    print(f"Coeficiente de determinación (R²): {r2}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Uso: python evaluate.py <input_file> <model_file> <metrics_file> <params_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    model_file = sys.argv[2]
    metrics_file = sys.argv[3]
    params_file = sys.argv[4]

    evaluate(input_file, model_file, metrics_file, params_file)
