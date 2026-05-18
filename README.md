# TFG — Predicción del *redshift* de LGRBs con Variational Quantum Regression

Este repositorio recoge el código del Trabajo de Fin de Grado en el que se implementa un regresor cuántico variacional de 7 qubits para predecir el *redshift* de Long Gamma-Ray Bursts (LGRBs) a partir de siete *features* observacionales del catálogo de Swift. La arquitectura del circuito es del tipo EfficientSU2 con *data re-uploading*, y la función de coste combina el error cuadrático medio habitual con un término de *ranking* que penaliza las inversiones de la relación física esperada entre la densidad de columna de hidrógeno y el *redshift*:

```
C_total(θ) = MSE(θ) + λ_NH · C_NH_ranking(θ)
```

El planteamiento metodológico (selección de *features*, criterios de limpieza, separación de SGRBs, etc.) sigue de cerca el de Narendra et al., con el que se compara el rendimiento.


## Contenido del repositorio

```
.
├── prep_datos_TFG.ipynb            Preparación del dataset
├── Circuito_cuantico.py            Entrenamiento y validación cruzada
├── GRAFÍCAS.ipynb                  Figuras finales de la memoria
│
├── combined_data_with_redshift_V8.csv   Catálogo crudo (251 GRBs)
├── LGRBs limpio def.csv                 Dataset tras la limpieza (225 LGRBs)
│
├── requirements.txt
├── .gitignore
└── README.md
```


## Datos

El dataset crudo, `combined_data_with_redshift_V8.csv`, contiene 251 GRBs con todas las columnas observacionales y valores faltantes. El pipeline de limpieza, implementado en `prep_datos_TFG.ipynb`, replica el procedimiento de Narendra et al. en tres pasos.

Primero se descartan los SGRBs aplicando el corte habitual `T90 ≥ 2 s`. A continuación, se marcan como NaN los valores físicamente inverosímiles (`log10NH < 20`, `Alpha > 3`, `Gamma > 3`, `PhotonIndex < 0`) y se imputan mediante MICE con un estimador `BayesianRidge` (20 iteraciones, semilla 42). Por último, se aplica un M-estimator robusto de Huber sobre la regresión polinómica de Narendra y se eliminan los GRBs cuyo peso queda por debajo de 0.65. El resultado son los 225 LGRBs de `LGRBs limpio def.csv`, con las siete *features* utilizadas en el modelo (`log10NH`, `log10PeakFlux`, `PhotonIndex`, `log10Ta`, `log10Fa`, `Gamma`, `Alpha`) y la variable objetivo `log Redshift = log10(z + 1)`. La transformación logarítmica del *redshift* estabiliza el ajuste numérico; el código deshace la transformación antes de calcular las métricas finales sobre `z`.


## Arquitectura del modelo

El circuito consta de 7 qubits, uno por *feature*. Cada capa repite la estructura de EfficientSU2: una codificación `Ry(x_i)` con *data re-uploading*, un bloque entrenable `Rz–Ry`, una cadena de CNOTs de izquierda a derecha, un segundo bloque `Rz–Ry`, una cadena de CNOTs en sentido inverso, y un tercer bloque `Rz–Ry`. La doble cadena de CNOTs con conectividad alternada maximiza la mezcla de información entre pares de qubits tras dos capas. El número de parámetros entrenables por capa es por tanto `6 × 7 = 42`.

Como observable se utiliza la media de los operadores `Z` sobre todos los qubits, y el cálculo se realiza con `StatevectorEstimator` (simulación exacta sobre CPU, sin ruido). La optimización de los parámetros se hace con L-BFGS-B de SciPy, calculando los gradientes de forma analítica mediante *parameter-shift rule* a través de `EstimatorQNN`.

La función de coste añade al MSE un término de *ranking* por pares, vectorizado con `np.triu_indices`, que penaliza las predicciones que invierten el orden esperado en `NH → z`. El peso relativo viene dado por la constante `LAMBDA_NH` al inicio del script (`0.05` por defecto; con `0.0` se recupera el MSE puro).

La validación se realiza por *cross-validation* repetido `10 × 10`, con búsqueda adaptativa del número de capas óptimo entre `LAYERS_MIN` y `LAYERS_MAX` y una paciencia de 1.


## Requisitos e instalación

El proyecto está probado con Python 3.10 y 3.11. La parte cuántica depende de `qiskit ≥ 1.0`, `qiskit-machine-learning` y `qiskit-algorithms`; el resto son dependencias científicas estándar (numpy, pandas, scipy, scikit-learn, matplotlib, statsmodels, seaborn) más Jupyter para los notebooks. La lista completa con las versiones recomendadas está en `requirements.txt`.

```bash
git clone https://github.com/<usuario>/<repo>.git
cd <repo>

python -m venv .venv
source .venv/bin/activate     # en Windows: .venv\Scripts\activate

pip install -r requirements.txt
```


## Ejecución

El flujo se divide en tres pasos. El primero, la preparación del dataset, se ejecuta abriendo `prep_datos_TFG.ipynb` y corriendo todas las celdas en orden; el notebook lee `combined_data_with_redshift_V8.csv` y guarda `LGRBs limpio def.csv` al final.

El segundo paso es el entrenamiento del modelo cuántico:

```bash
python Circuito_cuantico.py
```

Conviene tener en cuenta que el tiempo de ejecución no es despreciable: escala con `LAYERS_MAX × N_REPEATS_CV × N_FOLDS_CV` y, aunque el script paraleliza los *folds* mediante `joblib` (`N_JOBS = -1` por defecto), una ejecución completa en una máquina sin GPU puede llevar varias horas. Al finalizar se generan tres CSV de resultados (`vqr7q_layers_summary.csv`, `vqr7q_cv_detail.csv`, `vqr7q_predictions.csv`), un *log* completo de la ejecución (`vqr_su2_nh.log`) y ocho figuras de diagnóstico (curvas de entrenamiento, *scatter* y residuos del mejor *fold*, *boxplots* y *heatmap* por capa, y el diagnóstico físico NH→z).

El tercer paso es `GRAFÍCAS.ipynb`, que lee los CSV generados por el script y produce las figuras finales con el estilo de publicación que aparece en la memoria: distribuciones por capa en *violin plot*, *boxplots* de `r` y RMSE, *scatter* estilo Narendra con bandas 1σ y 2σ, y la distribución de residuos del modelo óptimo.


## Configuración

Los hiperparámetros relevantes están declarados como constantes al inicio de `Circuito_cuantico.py` y pueden modificarse sin tocar el resto del código:

```python
N_QUBITS      = 7
X_RANGE       = (0, np.pi)     # rango de codificación de las features
MAXITER       = 150            # iteraciones máximas de L-BFGS-B por fold
N_REPEATS_CV  = 10
N_FOLDS_CV    = 10
LAYERS_MIN    = 1
LAYERS_MAX    = 6
PATIENCE      = 1
MIN_DELTA     = 0.005
LAMBDA_NH     = 0.05           # peso del término NH-ranking (0 = MSE puro)
N_JOBS        = -1
```


## Reproducibilidad

Todas las semillas están fijadas a 42: la inicialización de los parámetros del circuito (`np.random.default_rng(seed=42)`), la imputación MICE y los `KFold` de la *cross-validation*. Como el *backend* es de simulación exacta y no introduce ruido cuántico, una misma versión del código en un mismo entorno produce resultados deterministas.


## Autor

Hugo Elche Asensio, Grado en Física, Universidad Europea de Valencia, curso 2025–2026.
Dirigido por Javier López Prieto.


## Referencias

- Narendra, A., Dainotti, M., Sarkar, M., Lenart, A., Bogdan, M., Pollo, A., Zhang, B., Rabeda, A., Petrosian, V., & Iwasaki, K. (2025). *Gamma-ray burst redshift estimation using machine learning and the associated web app*. Astronomy & Astrophysics, 698, A92.
- Qiskit Machine Learning, https://qiskit-community.github.io/qiskit-machine-learning/
- Swift GRB Catalogue, https://swift.gsfc.nasa.gov/results/batgrbcat/


## Licencia

Por defecto, el contenido del repositorio se distribuye sin licencia explícita y queda sujeto a los derechos del autor. Si se desea permitir reutilización, basta con añadir un archivo `LICENSE` con la licencia elegida (MIT o Apache-2.0 son las habituales para código académico).
