# Insatisfacción de clientes en Olist

Proyecto de Modelos y Simulación II, Universidad de Antioquia.

Predicción de clientes insatisfechos (calificación 1 o 2) en el [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) e identificación de los factores asociados.

## Contenido

| Ruta | Qué es |
|---|---|
| `01_eda_olist.ipynb` | Análisis exploratorio. Descarga los datos, construye la tabla por orden y exporta las figuras del informe |
| `informe/informe.tex` | Informe en plantilla IEEE (para Overleaf), con `referencias.bib` y `figuras/` |
| `informe/informe.pdf` | Informe compilado |
| `estado_del_arte.md` | Fichas de los artículos revisados |

## Cómo reproducir

Local (Python 3.12):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/jupyter notebook 01_eda_olist.ipynb
```

En Colab: abrir el notebook y descomentar la línea `!pip install -q kagglehub` de la primera celda.

El dataset se descarga con `kagglehub` sin credenciales. El notebook genera `data/olist_ordenes.csv.gz` (una fila por orden), que usarán los notebooks de modelado, y las figuras en `informe/figuras/`.

Para el informe en Overleaf: subir la carpeta `informe/` completa y compilar `informe.tex` con pdfLaTeX.
