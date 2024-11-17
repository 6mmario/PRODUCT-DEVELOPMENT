# Universidad Galileo
# Mario Obed Morales Guitz
## 24006981

---

# **Pipeline de Machine Learning con DVC**

Este repositorio implementa un pipeline reproducible y modular para entrenar, evaluar y versionar modelos de Machine Learning utilizando **DVC (Data Version Control)**. Este enfoque facilita el seguimiento de versiones, la reproducibilidad de experimentos y la colaboración en proyectos de ciencia de datos.

## **Índice**
1. [Descripción del Pipeline](#descripción-del-pipeline)
2. [Requisitos Previos](#requisitos-previos)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Configuración Inicial](#configuración-inicial)
5. [Ejecución del Pipeline](#ejecución-del-pipeline)
6. [Detalles de las Fases](#detalles-de-las-fases)
7. [Métricas de Evaluación](#métricas-de-evaluación)
8. [Contribución](#contribución)

---

## **Descripción del Pipeline**
El pipeline se organiza en tres fases principales:
1. **Preprocesamiento:** Limpieza y transformación del conjunto de datos.
2. **Entrenamiento (Train):** Entrenamiento del modelo con los datos preprocesados.
3. **Evaluación (Evaluate):** Evaluación del modelo entrenado y generación de métricas.

El archivo `dvc.yaml` gestiona la configuración del pipeline y define las dependencias entre etapas.

---

## **Requisitos Previos**

Antes de comenzar, asegúrate de tener instalados los siguientes componentes:

- **Python** >= 3.8  
- **DVC** >= 2.0  
- Dependencias adicionales:  
  Instala todas las librerías necesarias desde el archivo `requirements.txt`:

  ```bash
  pip install -r requirements.txt
  ```

---

## **Estructura del Proyecto**

La estructura del repositorio sigue esta organización:

```plaintext
├── data/                # Carpeta para almacenar datasets.
├── params.yaml          # Archivo de configuración para parámetros del pipeline.
├── src/                 # Código fuente para cada etapa del pipeline.
├── models/              # Carpeta donde se guardan los modelos entrenados.
├── metrics.json         # Métricas generadas durante la evaluación.
├── dvc.yaml             # Configuración principal del pipeline.
├── dvc.lock             # Archivo de control de dependencias del pipeline.
└── README.md            # Documentación del proyecto.
```

---

## **Configuración Inicial**

1. **Coloca el Dataset:**
   - Copia tu dataset inicial en la carpeta `data/`. Por ejemplo, el dataset debe estar en formato Parquet (`data/df_mflow_test.parquet`).

2. **Configura los Parámetros:**
   - Edita el archivo `params.yaml` para personalizar las características, la variable objetivo, y otros parámetros del pipeline:

     ```yaml
     train:
       test_size: 0.2
       random_state: 42
       alpha: 0.9

     preprocessing:
       target: Churn
       features:
         - MonthlyCharges
         - TotalCharges
         - Tenure
         - StreamingMovies
         - ContractType
         - PaymentMethod
     ```

3. **Inicializa DVC:**  
   Si aún no has inicializado un repositorio DVC, hazlo con el comando:  
   ```bash
   dvc init
   ```

4. **Configura los stages:**  
   Ejecuta los siguientes comandos para definir las etapas del pipeline:

   - **Preprocesamiento:**
     ```bash
     dvc stage add -n preprocess \
     -d data/df_mflow_test.parquet \
     -d params.yaml \
     -o data/clean_data.csv \
     python src/preprocess.py data/df_mflow_test.parquet data/clean_data.csv params.yaml
     ```

   - **Entrenamiento:**
     ```bash
     dvc stage add -n train \
     -d data/clean_data.csv \
     -d params.yaml \
     -o models/model.pkl \
     python src/train.py data/clean_data.csv models/model.pkl params.yaml
     ```

   - **Evaluación:**
     ```bash
     dvc stage add -n evaluate \
     -d data/clean_data.csv \
     -d models/model.pkl \
     -d params.yaml \
     -M metrics.json \
     python src/evaluate.py data/clean_data.csv models/model.pkl metrics.json params.yaml
     ```

---

## **Ejecución del Pipeline**

1. Para ejecutar todo el pipeline desde cero:  
   ```bash
   dvc repro
   ```

2. Para ejecutar una etapa específica del pipeline:  
   ```bash
   dvc repro <stage_name>
   ```
   Por ejemplo:  
   ```bash
   dvc repro train
   ```

3. Para verificar el estado de las dependencias y outputs:  
   ```bash
   dvc status
   ```

---

## **Detalles de las Fases**

### **1. Preprocesamiento**
Limpieza y transformación de los datos. Produce un dataset limpio en formato CSV.  
**Comando:**  
```bash
python src/preprocess.py data/df_mflow_test.parquet data/clean_data.csv params.yaml
```

### **2. Entrenamiento**
Entrena el modelo utilizando los datos preprocesados. Guarda el modelo entrenado en `models/model.pkl`.  
**Comando:**  
```bash
python src/train.py data/clean_data.csv models/model.pkl params.yaml
```

### **3. Evaluación**
Evalúa el modelo entrenado en el conjunto de prueba y genera métricas en formato JSON.  
**Comando:**  
```bash
python src/evaluate.py data/clean_data.csv models/model.pkl metrics.json params.yaml
```

---

## **Métricas de Evaluación**

Las métricas se guardan en un archivo `metrics.json`. Ejemplo de métricas generadas:

```json
{
    "mse": 0.01875,  // Error cuadrático medio
    "r2": 0.87234,   // Coeficiente de determinación
    "mae": 0.01543   // Error absoluto medio
}
```

Puedes visualizar las métricas almacenadas ejecutando:
```bash
dvc metrics show
```

---

## **Contribución**

Si deseas contribuir al proyecto:
1. Haz un fork del repositorio.
2. Crea una nueva rama:
   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```
3. Realiza tus cambios y realiza un commit:
   ```bash
   git commit -m "Agrega nueva funcionalidad"
   ```
4. Sube tus cambios:
   ```bash
   git push origin feature/nueva-funcionalidad
   ```
5. Crea un Pull Request.

---