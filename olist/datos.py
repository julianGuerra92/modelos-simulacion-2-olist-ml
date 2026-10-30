"""Carga de la tabla por orden y partición entrenamiento/prueba."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RUTA_DATOS = "data/olist_ordenes.csv.gz"
SEMILLA = 42
FRACCION_PRUEBA = 0.2

NUM = ["n_articulos", "n_productos", "n_vendedores", "precio_total", "flete_total", "flete_relativo",
       "valor_pagado", "cuotas", "n_medios_pago", "peso_total_g", "volumen_total_cm3", "fotos_promedio",
       "largo_descripcion", "largo_nombre", "dias_aprobacion", "dias_a_transportadora", "dias_entrega",
       "dias_estimados", "retraso_dias", "distancia_km", "mes_compra", "dia_semana_compra", "hora_compra"]
BIN = ["entrega_tarde", "vendedor_tarde", "mismo_estado"]
CAT = ["tipo_pago", "categoria", "customer_state", "seller_state"]
OBJ = "insatisfecho"
ENTRADAS = NUM + BIN + CAT


def cargar(ruta=RUTA_DATOS):
    """Lee la tabla generada por 01_eda_olist.ipynb (una fila por orden entregada)."""
    datos = pd.read_csv(ruta, index_col="order_id")
    datos[NUM] = datos[NUM].replace([np.inf, -np.inf], np.nan)
    return datos


def particion(datos, fraccion_prueba=FRACCION_PRUEBA, semilla=SEMILLA):
    """Separa un conjunto de prueba estratificado que no se usa hasta la evaluación final."""
    X, y = datos[ENTRADAS], datos[OBJ].to_numpy()
    return train_test_split(X, y, test_size=fraccion_prueba, stratify=y, random_state=semilla)
