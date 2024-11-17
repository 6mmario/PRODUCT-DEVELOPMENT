# src/train.py

import sys
import yaml
import pandas as pd
import joblib
import optuna
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def train(input_file, model_file, params_file):
    """
    Entrena un modelo de regresión utilizando Optuna para optimizar los hiperparámetros.
    
    Args:
        input_file (str): Ruta al archivo CSV con los datos preprocesados.
        model_file (str): Ruta para guardar el modelo entrenado.
        params_file (str): Ruta al archivo YAML con la configuración de hiperparámetros y características.
    """
    # Cargar el dataset preprocesado
    df = pd.read_csv(input_file)
    print(f"Columnas en el archivo de entrada: {df.columns.tolist()}")  # Validación de columnas

    # Leer los parámetros de configuración
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    # Validar que todas las características existan en el dataset
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Las siguientes columnas faltan en el archivo de entrada: {missing_features}")

    # Separar características y objetivo
    X = df[features]
    y = df[target]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
    )

    # Función objetivo para la optimización de hiperparámetros
    def objective(trial):
        # Seleccionar modelo
        model_name = trial.suggest_categorical("model", ["LinearRegression", "RandomForest", "GradientBoosting"])

        if model_name == "LinearRegression":
            model = LinearRegression()
        elif model_name == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
            max_depth = trial.suggest_int("max_depth", 10, 50, step=10)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=-1)
        elif model_name == "GradientBoosting":
            learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.1)
            n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
            model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, random_state=0)

        # Crear pipeline con escalado y modelo
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        # Entrenar el modelo
        pipeline.fit(X_train, y_train)

        # Evaluar el modelo en el conjunto de prueba
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    # Crear estudio de Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    # Mostrar los resultados del estudio
    print(f"Mejores hiperparámetros: {study.best_params}")
    print(f"Mejor MSE: {study.best_value}")

    # Configurar el mejor modelo según los hiperparámetros encontrados
    best_params = study.best_params
    if best_params['model'] == "LinearRegression":
        best_model = LinearRegression()
    elif best_params['model'] == "RandomForest":
        best_model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            random_state=0,
            n_jobs=-1
        )
    elif best_params['model'] == "GradientBoosting":
        best_model = GradientBoostingRegressor(
            learning_rate=best_params['learning_rate'],
            n_estimators=best_params['n_estimators'],
            random_state=0
        )

    # Crear pipeline final con el mejor modelo
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', best_model)
    ])
    best_pipeline.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(best_pipeline, model_file)
    print(f"Modelo guardado en: {model_file}")

    # Visualizar los resultados del estudio de Optuna
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Curva de convergencia de los hiperparámetros")
    plt.show()

    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Importancia de los hiperparámetros")
    plt.show()

    # Evaluar el modelo en el conjunto de prueba
    y_pred = best_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Error cuadrático medio (MSE) del mejor modelo: {mse}")
    print(f"Coeficiente de determinación (R²) del mejor modelo: {r2}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python train.py <input_file> <model_file> <params_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    model_file = sys.argv[2]
    params_file = sys.argv[3]

    train(input_file, model_file, params_file)
