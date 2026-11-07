"""Transformaciones de entrada. Todo se ajusta solo con los datos de entrenamiento de cada pliegue."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .datos import BIN, CAT, NUM

# Variables con cola larga a la derecha (sección 5 del EDA): se aplica log(1 + x) antes de estandarizar
ASIMETRICAS = ["precio_total", "flete_total", "flete_relativo", "valor_pagado", "peso_total_g",
               "volumen_total_cm3", "largo_descripcion", "dias_aprobacion", "dias_a_transportadora",
               "distancia_km"]
SIMETRICAS = [v for v in NUM if v not in ASIMETRICAS]

# 15 niveles más frecuentes en entrenamiento; el resto (y los niveles nuevos en prueba) van a "Otros"
MAX_NIVELES = 15


def _log_sin_negativos(X):
    # Los días negativos son errores de registro (1,4 % de las órdenes): se recortan a cero
    return np.log1p(np.clip(X, 0, None))


def preprocesador(max_niveles=MAX_NIVELES):
    asimetricas = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(_log_sin_negativos, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    simetricas = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categoricas = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="desconocido"),
        OneHotEncoder(max_categories=max_niveles + 1, handle_unknown="infrequent_if_exist",
                      sparse_output=False),
    )
    return ColumnTransformer([
        ("asim", asimetricas, ASIMETRICAS),
        ("num", simetricas, SIMETRICAS),
        ("bin", "passthrough", BIN),
        ("cat", categoricas, CAT),
    ], verbose_feature_names_out=False)
