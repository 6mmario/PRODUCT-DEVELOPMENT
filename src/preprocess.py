# src/preprocess.py

import sys
import yaml
import pandas as pd


def preprocess(input_file, output_file, features, target):
    """
    Preprocesa un archivo .parquet seleccionando características específicas y la variable objetivo.
    
    Args:
        input_file (str): Ruta al archivo .parquet de entrada.
        output_file (str): Ruta para guardar el archivo CSV resultante.
        features (list): Lista de nombres de columnas que se usarán como características.
        target (str): Nombre de la columna objetivo.
    """
    # Cargar el dataset desde un archivo Parquet
    df = pd.read_parquet(input_file, engine='pyarrow')
    print(f"Columnas en el archivo de entrada: {df.columns.tolist()}")  # Ver columnas disponibles

    # Validar que las columnas requeridas existan en el dataset
    missing_columns = [col for col in features + [target] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Las siguientes columnas no están en el archivo de entrada: {missing_columns}")

    print(f"Se seleccionan las características: {features}")
    print(f"Se selecciona la columna objetivo: {target}")

    # Filtrar las columnas necesarias
    selected_columns = features + [target]
    df_selected = df[selected_columns]

    # Guardar el dataset filtrado como archivo CSV
    df_selected.to_csv(output_file, index=False, header=True)
    print(f"Preprocesamiento completado. Datos guardados en {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python preprocess.py <input_file> <output_file> <params_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]

    # Cargar configuración desde el archivo YAML
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    # Ejecutar la función de preprocesamiento
    preprocess(input_file, output_file, features, target)
