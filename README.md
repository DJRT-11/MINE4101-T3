# MINE4101-T3
Repositorio público para Taller 3 de Ciencia de Datos Aplicada

## Los resultados a las preguntas 1 - 3 del enunciado se encuentran en el notebook training/T3 Model Training.ipynb
## Los resultados a la pregunta 4 del enunciado se encuentra en la implementación del modelo a traves del presente repositorio
## Los resultados a la pregunta 5 del enunciado se encuentra en el notebook a-b_testing.ipynb

## Instrucciones previas

### Creación de entorno e instalación de paquetes
Para ejecutar este programa correctamente, se recomienda crear un entorno virtual de Python que contenga todas las librerías requeridas.

1. Instalar ``virtualenv`` en su instancia de Python: ``pip install virtualenv``
2. Crear un nuevo entorno en la carpeta de su preferencia: ``python -m venv nombre_entorno``
    (Puede cambiar ``nombre_entorno`` por el que prefiera)
3. Activar el entorno creado
    a. En Windows: ``nombre_entorno\Scripts\activate``
    b. En Linux: ``source nombre_entorno/bin/activate``
4. Instalar el archivo de dependencias
    a. Descargue o clone este repositorio (en caso de haberlo descargado, debe descomprimirlo)
    b. ``pip install -r dirección\donde\descargó\el\repositorio\requirements.txt``

### Inicialización y prueba
1. Active el entorno creado previamente
    a. En Windows: ``nombre_entorno\Scripts\activate``
    b. En Linux: ``source nombre_entorno/bin/activate``
2. Diríjase a la carpeta donde descargó/clonó el repositorio
3. Inicie el servidor: ``uvicorn main:app --reload``
4. Utilice una herramienta como Postman para utilizar las funciones POST

## Descripción de estructura y archivos

### Archivos
- main.py: app principal. Contiene las funciones llamadas a través de herramientas de despliegue como Postman.
- prediction_model.py: define las clases y funciones que hacen referencia a los modelos de clasificación entrenados.
- data_model.py: define la estructura de características utilizadas por los modelos para realizar predicciones.
- requirements.txt: contiene las librerías necesarias para el funcionamiento del programa.
- churn_future_random.ipynb: describe la separación del conjunto de datos futuros en dos grupos iguales, <i>baseline</i> y <i>optimizado</i> (50% en cada grupo), y en 7 subgrupos (uno para cada día de la semana). Jupyter Notebook.
- a-b_testing.ipynb: contiene las operaciones realizadas para determinar los resultados de la sección "A/B Testing" del enunciado. Jupyter Notebook.

### Carpetas

#### a-b_testing
Contiene los datos divididos en los grupos necesarios para la sección "A/B Testing" del enunciado.
#####   input-data
Conjuntos de datos separados en <i>baseline</i> y <i>optimizado</i>, y en un subgrupo para cada día de la semana (01 a 07)
#####   outputs
Conjuntos de archivos de predicción para cada archivo en ``input data``.

#### models
Contiene archivos .joblib con los modelos de predicción.
- churn-baseline-v1.0.joblib: Modelo de regresión logística, sin optimizar.
- churn-v1.0.joblib: Modelo de Random Forest, con hiperparámetros optimizados a través de validación cruzada.

#### training
Información relacionada al entrenamiento de los modelos
- data: datos en formato JSON utilizados en el proceso de entrenamiento.
- functions: functiones utilizadas para transformación y preprocesamiento de datos ingresados en el endpoint.

