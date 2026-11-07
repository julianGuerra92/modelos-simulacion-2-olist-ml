# Insatisfacción de clientes en Olist

Proyecto de Modelos y Simulación II, Universidad de Antioquia.

Predicción de clientes insatisfechos (calificación 1 o 2) en el [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) e identificación de los factores asociados. El modelo se aplica cuando se registra la entrega de la orden, antes de que el cliente califique.

## Contenido

| Ruta | Qué es |
|---|---|
| `01_eda_olist.ipynb` | Análisis exploratorio. Descarga los datos, construye la tabla por orden, analiza las calificaciones de 3 estrellas y exporta las figuras de la exploración |
| `02_modelos.ipynb` | Sección 4 de la guía: configuración experimental, búsqueda en malla de seis modelos y comparación en entrenamiento, validación y prueba |
| `03_reduccion_dimension.ipynb` | Sección 5 de la guía: análisis individual de variables, PCA y UMAP sobre los dos mejores modelos |
| `olist/` | Código que comparten los notebooks 02 y 03: carga y partición (`datos.py`), preprocesamiento (`preprocesamiento.py`), métricas e intervalos de confianza (`evaluacion.py`) |
| `resultados/` | Configuración y métricas de los modelos (`seccion4.json`, `seccion5.csv`) |
| `informe/` | Informe en plantilla IEEE (`informe.tex`, `referencias.bib`, `figuras/`) y su PDF compilado (`Proyecto_Modelos_II.pdf`) |
| `estado_del_arte.md` | Fichas de los artículos revisados |

## Cómo reproducir

Local (Python 3.12):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/jupyter notebook
```

Los notebooks se ejecutan en orden, desde la raíz del repositorio:

1. `01_eda_olist.ipynb` descarga el dataset con `kagglehub` (no pide credenciales) y genera `data/olist_ordenes.csv.gz`, una fila por orden.
2. `02_modelos.ipynb` lee esa tabla, ajusta los seis modelos y guarda `resultados/seccion4.json`. Con 12 núcleos tarda cerca de una hora; las mallas del bosque aleatorio y de la red neuronal son las más lentas.
3. `03_reduccion_dimension.ipynb` toma de `resultados/seccion4.json` los dos mejores modelos y los evalúa con selección de variables, PCA y UMAP.

Las figuras del informe se escriben en `informe/figuras/`. La partición y los modelos usan la semilla 42, así que los resultados se repiten de una ejecución a otra.

En Colab: clonar el repositorio, entrar a su carpeta y descomentar las líneas de instalación de la primera celda de cada notebook.

Para el informe en Overleaf: subir la carpeta `informe/` completa y compilar `informe.tex` con pdfLaTeX.
