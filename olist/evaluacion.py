"""Métricas, búsqueda en malla con validación cruzada e intervalos de confianza.

Métrica principal: F1 de la clase insatisfecha con el umbral de decisión que la maximiza.
Durante la búsqueda en malla el umbral se elige dentro de cada pliegue de validación; para la
evaluación final se fija con las predicciones fuera de pliegue del entrenamiento y se aplica tal
cual al conjunto de prueba.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             make_scorer, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from .datos import SEMILLA

N_PLIEGUES = 5
RESPUESTA = ("predict_proba", "decision_function")
METRICAS = {"f1": "F1", "auc_roc": "AUC-ROC", "auc_pr": "AUC-PR", "sensibilidad": "Sensibilidad",
            "precision": "Precisión", "exactitud_bal": "Exactitud bal."}


def pliegues(n=N_PLIEGUES, semilla=SEMILLA):
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=semilla)


def umbral_f1(y, s):
    """Umbral sobre el puntaje que maximiza el F1 de la clase positiva."""
    p, r, u = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return u[np.argmax(f1)]


def metricas(y, s, umbral):
    pred = (s >= umbral).astype(int)
    return {
        "f1": f1_score(y, pred, zero_division=0),
        "auc_roc": roc_auc_score(y, s),
        "auc_pr": average_precision_score(y, s),
        "sensibilidad": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "exactitud_bal": balanced_accuracy_score(y, pred),
    }


def _metrica_con_umbral_propio(nombre):
    return lambda y, s: metricas(y, s, umbral_f1(y, s))[nombre]


PUNTAJES = {k: make_scorer(_metrica_con_umbral_propio(k), response_method=RESPUESTA) for k in METRICAS}


def puntaje(modelo, X):
    """Probabilidad de insatisfacción, o la función de decisión si el modelo no da probabilidades."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def buscar(modelo, malla, X, y, cv=None, n_jobs=-1):
    """Búsqueda en malla con validación cruzada estratificada; reajusta con la mejor F1."""
    gs = GridSearchCV(modelo, malla, scoring=PUNTAJES, refit="f1", cv=cv or pliegues(),
                      return_train_score=True, n_jobs=n_jobs)
    return gs.fit(X, y)


def semiancho_ic(valores, nivel=0.95):
    """Semiancho del intervalo de confianza t de Student para la media de los pliegues."""
    valores = np.asarray(valores)
    return stats.t.ppf((1 + nivel) / 2, len(valores) - 1) * valores.std(ddof=1) / np.sqrt(len(valores))


def tabla_malla(gs, metrica="f1"):
    """Una fila por configuración: entrenamiento y validación con su intervalo de confianza."""
    cv = gs.cv_results_
    n = gs.n_splits_
    filas = []
    for i, params in enumerate(cv["params"]):
        fila = {k.split("__")[-1]: v for k, v in params.items()}
        for parte in ("train", "test"):
            folds = [cv[f"split{j}_{parte}_{metrica}"][i] for j in range(n)]
            etiqueta = "entrenamiento" if parte == "train" else "validación"
            fila[etiqueta] = np.mean(folds)
            fila[f"± {etiqueta}"] = semiancho_ic(folds)
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("validación", ascending=False).reset_index(drop=True)


def resumen_validacion(gs):
    """Media e intervalo de confianza de cada métrica en validación para la mejor configuración."""
    cv, i, n = gs.cv_results_, gs.best_index_, gs.n_splits_
    return {k: (np.mean(f := [cv[f"split{j}_test_{k}"][i] for j in range(n)]), semiancho_ic(f))
            for k in METRICAS}


def umbral_fuera_de_pliegue(modelo, X, y, cv=None, n_jobs=-1):
    """Elige el umbral con predicciones fuera de pliegue del conjunto de entrenamiento."""
    metodo = "predict_proba" if hasattr(modelo, "predict_proba") else "decision_function"
    s = cross_val_predict(modelo, X, y, cv=cv or pliegues(), method=metodo, n_jobs=n_jobs)
    s = s[:, 1] if s.ndim == 2 else s
    return umbral_f1(y, s)


def evaluar_prueba(modelo, umbral, X, y, n_boot=1000, semilla=SEMILLA):
    """Métricas en prueba con el umbral fijado en entrenamiento e IC del 95 % por bootstrap."""
    s = puntaje(modelo, X)
    puntual = metricas(y, s, umbral)
    rng = np.random.default_rng(semilla)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots.append(metricas(y[i], s[i], umbral))
    boots = pd.DataFrame(boots)
    return {k: (v, boots[k].quantile(0.025), boots[k].quantile(0.975)) for k, v in puntual.items()}


def resumen_modelo(gs, X, y, X_prueba, y_prueba):
    """Mejor configuración de una malla con sus métricas de entrenamiento, validación y prueba."""
    cv, i, n = gs.cv_results_, gs.best_index_, gs.n_splits_
    entrenamiento = {k: np.mean([cv[f"split{j}_train_{k}"][i] for j in range(n)]) for k in METRICAS}
    umbral = umbral_fuera_de_pliegue(gs.best_estimator_, X, y)
    return {
        "parametros": {k.split("__")[-1]: v for k, v in gs.best_params_.items()},
        "umbral": float(umbral),
        "entrenamiento": entrenamiento,
        "validacion": resumen_validacion(gs),
        "prueba": evaluar_prueba(gs.best_estimator_, umbral, X_prueba, y_prueba),
        "modelo": gs.best_estimator_,
    }


def tabla_resultados(resultados, metrica="f1"):
    """Compara modelos en entrenamiento, validación (IC t) y prueba (IC bootstrap)."""
    filas = []
    for nombre, r in resultados.items():
        v, p = r["validacion"][metrica], r["prueba"][metrica]
        filas.append({"modelo": nombre, "entrenamiento": round(r["entrenamiento"][metrica], 3),
                      "validación": formato_ic(*v), "prueba": formato_ic(p[0], bajo=p[1], alto=p[2])})
    return pd.DataFrame(filas).set_index("modelo")


def formato_ic(media, semiancho=None, bajo=None, alto=None):
    if semiancho is not None:
        return f"{media:.3f} ± {semiancho:.3f}"
    return f"{media:.3f} [{bajo:.3f}, {alto:.3f}]"
