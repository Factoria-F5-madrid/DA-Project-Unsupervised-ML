# 🧠 Taller de Aprendizaje Automático **No Supervisado**

Este repositorio contiene un taller práctico de **machine learning no supervisado**: reducción de dimensionalidad (**PCA**, **t-SNE**), **clustering** (K-Means, Aglomerativo, GMM, DBSCAN) y **detección de anomalías** (Isolation Forest).

La novedad de esta versión es que el taller se divide en **dos notebooks complementarios**, cada uno con su propio dataset. La gracia está en el **contraste entre ambos**: un mismo conjunto de técnicas se comporta de forma muy distinta según el tipo de datos y según si tenemos o no una etiqueta de referencia.

---

## 📥 Antes de empezar: descarga los datos

Los datasets **no están en el repositorio**. Descárgalos y colócalos en `data/`:

| Fichero | Origen | Tamaño |
|---|---|---|
| `data/mushrooms.csv` | [UCI 73](https://archive.ics.uci.edu/dataset/73/mushroom) · [espejo en Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification) | 366 KB |
| `data/credit_card.csv` | [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) | 882 KB |

**Atajo:** ambos ya preparados en la [carpeta de formación en Drive](https://drive.google.com/drive/folders/1apSXjn6eQ5o9RdutbD4skSjvH6ytvR06?usp=sharing).

Ver también `data/LEEME.txt`.

---

## 🗂️ Estructura del repositorio

| Notebook | Dataset | Tipo de datos | ¿Hay etiqueta? |
|---|---|---|---|
| [`workshop-clustering-Mushrooms.ipynb`](workshop-clustering-Mushrooms.ipynb) | `data/mushrooms.csv` | Categóricos | Sí — `class` (solo para **validar**) |
| [`workshop-clustering-creditcard.ipynb`](workshop-clustering-creditcard.ipynb) | `data/credit_card.csv` | Numéricos | No — segmentación **de verdad** |

> Hay que entregar **los dos notebooks**. No son independientes: la Parte 2 da por sabido lo aprendido en la Parte 1.

---

## 🍄 Parte 1 — Setas (datos categóricos, *con* etiqueta)

**Notebook:** [`workshop-clustering-Mushrooms.ipynb`](workshop-clustering-Mushrooms.ipynb) · **Dataset:** `data/mushrooms.csv`
🔗 [Mushroom Dataset (Kaggle)](https://www.kaggle.com/uciml/mushroom-classification) · [UCI](https://archive.ics.uci.edu/ml/datasets/Mushroom)

Cada fila es un hongo descrito con **~22 variables, todas categóricas** (forma, color, olor, etc.). La variable `class` es **binaria**: `e` (comestible) / `p` (venenoso).

La clave pedagógica: **tenemos etiqueta, pero el clustering NO la usa**. La reservamos *solo para validar* a posteriori cuánta estructura real ha recuperado el modelo sin haberla visto.

**Qué se trabaja:**
- Carga, EDA y detección de nulos encubiertos (el valor `'?'`) y de columnas constantes (`veil-type`).
- Imputación con la moda y **One-Hot Encoding** (`pd.get_dummies`).
- **PCA** y **t-SNE** para visualizar un dataset de >100 dimensiones en 2D.
- **Random Forest** como *línea base supervisada* (¿cuánta información hay realmente?) y estudio de cuántas componentes PCA bastan para mantener la precisión.
- **Clustering**: K-Means (codo + *silhouette*), Aglomerativo (con **dendrograma**), GMM y DBSCAN.
- Lección con **DBSCAN**: con datos categóricos one-hot, la **distancia importa** (euclídea vs **Jaccard**).
- Validación con etiqueta: **Adjusted Rand Index (ARI)** y **NMI**.
- **Isolation Forest** para detección de anomalías.

---

## 💳 Parte 2 — Tarjetas de crédito (datos numéricos, *sin* etiqueta)

**Notebook:** [`workshop-clustering-creditcard.ipynb`](workshop-clustering-creditcard.ipynb) · **Dataset:** `data/credit_card.csv`
🔗 [Credit Card Dataset for Clustering (Kaggle)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

Comportamiento de uso de ~9.000 titulares de tarjeta durante 6 meses, con **17 variables numéricas** (saldo, compras, adelantos de efectivo, límite, pagos…).

La clave pedagógica: **aquí NO hay etiqueta**. Es aprendizaje no supervisado «de verdad»: no se puede calcular ARI porque no existe una verdad de referencia. El éxito se mide con **métricas internas** y, sobre todo, con la **interpretabilidad** de los segmentos. El objetivo es **segmentar clientes** para una estrategia de marketing.

**Qué se trabaja:**
- Carga, EDA y tratamiento de nulos (imputación con la **mediana**, más robusta en datos sesgados).
- Observación del **sesgo** típico de datos financieros (histogramas).
- **Escalado** (`StandardScaler`) — imprescindible cuando las variables tienen escalas muy distintas.
- **PCA**: varianza explicada acumulada (*scree plot*) y proyección a 2D (aquí los datos son una **nube continua**, no grupos separados).
- **Clustering**: K-Means (codo + *silhouette*), Aglomerativo (dendrograma), GMM y DBSCAN.
- Validación **sin etiqueta**: *silhouette*, *Davies-Bouldin* y *Calinski-Harabasz*.
- Visualización de los segmentos con **t-SNE**.
- **Interpretación de perfiles** (heatmap de medias por cluster) → nombrar los segmentos en términos de negocio (VIP, riesgo, poco activos…). **Este es el entregable del caso.**
- **Isolation Forest** para detectar clientes atípicos.

---

## 🧩 ¿Por qué dos datasets?

| | 🍄 Setas | 💳 Tarjetas |
|---|---|---|
| Variables | Categóricas | Numéricas |
| Preprocesado clave | One-Hot Encoding | Escalado / imputación |
| Etiqueta | Sí (solo validar) | **No** |
| Cómo se valida | ARI / NMI (vs etiqueta) | Métricas internas + interpretabilidad |
| Estructura | Grupos separables | Nube continua |
| Distancia | Jaccard > euclídea | Euclídea sobre datos escalados |

**Conclusión transversal:** no hay un algoritmo ni una métrica que gane siempre. El acierto está en elegir el preprocesado, la distancia y la forma de validar **según el tipo de datos y el problema**.

---

## 🔧 Tecnologías

- Python · Pandas · NumPy
- Seaborn · Matplotlib (y opcionalmente Plotly)
- Scikit-learn: `PCA`, `TSNE`, `KMeans`, `AgglomerativeClustering`, `GaussianMixture`, `DBSCAN`, `IsolationForest`, `RandomForestClassifier` y métricas de clustering
- *(Opcional, para ir más allá)* `umap-learn`, `hdbscan`, `mlxtend`, SciPy (`linkage` / `dendrogram`)

---

## 📊 Evaluación

Se evaluarán las siguientes competencias **en ambos notebooks**:

**Competencia: Evaluar conjuntos de datos con herramientas de análisis y visualización**
- ✅ Uso y gestión de formato `.csv`
- ✅ Limpieza y preprocesado de datos
- ✅ Visualización de datos (Seaborn, Matplotlib, Plotly)
- ✅ Análisis exploratorio detallado (EDA)
- ✅ Técnicas de preprocesado (normalización, escalado, label/one-hot encoding)
- ✅ Técnicas avanzadas de limpieza (atípicos, imputación de faltantes)
- ✅ Técnicas de reducción de dimensionalidad (PCA, t-SNE)

**Competencia: Aplicar algoritmos de ML según el problema**
- ✅ Seleccionar las variables útiles y descartar las que no aportan
- ✅ Reconocer un caso de aprendizaje no supervisado
- ✅ Aplicar modelos de clustering
- ✅ Distinguir regresión / clasificación / clustering
- ✅ Separación de datos en train/test (Parte 1)
- ✅ Uso de modelos de *ensemble* (RandomForest como baseline en la Parte 1)
- ✅ Interpretación y validación de los resultados (ARI/NMI y métricas internas)

---

## 🚀 Cómo empezar

1. Clona el repositorio.
2. Los datasets están en la carpeta [`data/`](data/) (`data/mushrooms.csv` y `data/credit_card.csv`); al leerlos usa esa ruta, p. ej. `pd.read_csv("data/mushrooms.csv")`.
3. Abre cada notebook (Jupyter / VS Code) y **completa las celdas marcadas con comentarios** (`# ...`). Las celdas traen pistas, no la solución.
4. Empieza por la **Parte 1 (setas)** y luego haz la **Parte 2 (tarjetas)**.

> 💡 Cada notebook termina con una sección **«Para ir más allá»** con extensiones opcionales (UMAP, HDBSCAN, reglas de asociación, ingeniería de KPIs…) para quien quiera profundizar.
